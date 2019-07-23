# -*- coding: utf-8 -*-
import os
import copy
import time
import shutil

from nipype import config
from nipype import logging
import nipype.pipeline.engine as pe

import CPAC
from CPAC.utils.datasource import (
    create_anat_datasource,
    create_check_for_s3_node
)

from CPAC.anat_preproc.anat_preproc import create_anat_preproc
from CPAC.anat_preproc.longitudinal_preproc import template_creation_flirt

from CPAC.utils import Strategy, find_files

from CPAC.utils.utils import (
    create_log,
    check_config_resources,
    check_system_deps
)

logger = logging.getLogger('nipype.workflow')


def create_log_node(workflow, logged_wf, output, index, scan_id=None):
    try:
        log_dir = workflow.config['logging']['log_directory']
        if logged_wf:
            log_wf = create_log(wf_name='log_%s' % logged_wf.name)
            log_wf.inputs.inputspec.workflow = logged_wf.name
            log_wf.inputs.inputspec.index = index
            log_wf.inputs.inputspec.log_dir = log_dir
            workflow.connect(logged_wf, output, log_wf, 'inputspec.inputs')
        else:
            log_wf = create_log(wf_name='log_done_%s' % scan_id,
                                scan_id=scan_id)
            log_wf.base_dir = log_dir
            log_wf.inputs.inputspec.workflow = 'DONE'
            log_wf.inputs.inputspec.index = index
            log_wf.inputs.inputspec.log_dir = log_dir
            log_wf.inputs.inputspec.inputs = log_dir
            return log_wf
    except Exception as e:
        print(e)


def init_subject_wf(sub_dict, conf):
    c = copy.copy(conf)

    subject_id = sub_dict['subject_id']
    if sub_dict['unique_id']:
        subject_id += "_" + sub_dict['unique_id']

    log_dir = os.path.join(c.logDirectory, 'pipeline_%s' % c.pipelineName,
                           subject_id)
    if not os.path.exists(log_dir):
        os.makedirs(os.path.join(log_dir))

    config.update_config({
        'logging': {
            'log_directory': log_dir,
            'log_to_file': bool(getattr(c, 'run_logging', True))
        }
    })

    logging.update_logging(config)

    # Start timing here
    pipeline_start_time = time.time()
    # TODO LONG_REG change prep_worflow to use this attribute instead of the local var
    c.update('pipeline_start_time', pipeline_start_time)

    # Check pipeline config resources
    sub_mem_gb, num_cores_per_sub, num_ants_cores = \
        check_config_resources(c)

    # TODO LONG_REG understand and handle that
    # if plugin_args:
    #     plugin_args['memory_gb'] = sub_mem_gb
    #     plugin_args['n_procs'] = num_cores_per_sub
    # else:
    #     plugin_args = {'memory_gb': sub_mem_gb, 'n_procs': num_cores_per_sub}

    # perhaps in future allow user to set threads maximum
    # this is for centrality mostly
    # import mkl
    numThreads = '1'
    os.environ['OMP_NUM_THREADS'] = '1'  # str(num_cores_per_sub)
    os.environ['MKL_NUM_THREADS'] = '1'  # str(num_cores_per_sub)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(num_ants_cores)

    # calculate maximum potential use of cores according to current pipeline
    # configuration
    max_core_usage = int(c.maxCoresPerParticipant) * int(
        c.numParticipantsAtOnce)

    information = """

        C-PAC version: {cpac_version}

        Setting maximum number of cores per participant to {cores}
        Setting number of participants at once to {participants}
        Setting OMP_NUM_THREADS to {threads}
        Setting MKL_NUM_THREADS to {threads}
        Setting ANTS/ITK thread usage to {ants_threads}
        Maximum potential number of cores that might be used during this run: {max_cores}

    """

    logger.info(information.format(
        cpac_version=CPAC.__version__,
        cores=c.maxCoresPerParticipant,
        participants=c.numParticipantsAtOnce,
        threads=numThreads,
        ants_threads=c.num_ants_threads,
        max_cores=max_core_usage
    ))

    # Check system dependencies
    check_system_deps(check_ants='ANTS' in c.regOption,
                      check_ica_aroma='1' in str(c.runICA[0]))

    # absolute paths of the dirs
    c.workingDirectory = os.path.abspath(c.workingDirectory)
    if 's3://' not in c.outputDirectory:
        c.outputDirectory = os.path.abspath(c.outputDirectory)

    # Workflow setup
    workflow_name = 'resting_preproc_' + str(subject_id)
    workflow = pe.Workflow(name=workflow_name)
    workflow.base_dir = c.workingDirectory
    workflow.config['execution'] = {
        'hash_method': 'timestamp',
        'crashdump_dir': os.path.abspath(c.crashLogDirectory)
    }

    # Extract credentials path if it exists
    try:
        creds_path = sub_dict['creds_path']
        if creds_path and 'none' not in creds_path.lower():
            if os.path.exists(creds_path):
                input_creds_path = os.path.abspath(creds_path)
            else:
                err_msg = 'Credentials path: "%s" for subject "%s" was not ' \
                          'found. Check this path and try again.' % (
                              creds_path, subject_id)
                raise Exception(err_msg)
        else:
            input_creds_path = None
    except KeyError:
        input_creds_path = None

    # TODO ASH normalize file paths with schema validator
    template_anat_keys = [
        ("anat", "template_brain_only_for_anat"),
        ("anat", "template_skull_for_anat"),
        ("anat", "ref_mask"),
        ("anat", "template_symmetric_brain_only"),
        ("anat", "template_symmetric_skull"),
        ("anat", "dilated_symmetric_brain_mask"),
        ("anat", "templateSpecificationFile"),
        ("anat", "lateral_ventricles_mask"),
        ("anat", "PRIORS_CSF"),
        ("anat", "PRIORS_GRAY"),
        ("anat", "PRIORS_WHITE"),
        ("other", "configFileTwomm"),
    ]

    for key_type, key in template_anat_keys:
        node = create_check_for_s3_node(
            key,
            getattr(c, key), key_type,
            input_creds_path, c.workingDirectory
        )

        setattr(c, key, node)

    if c.reGenerateOutputs is True:
        working_dir = os.path.join(c.workingDirectory, workflow_name)
        erasable = list(find_files(working_dir, '*sink*')) + \
                   list(find_files(working_dir, '*link*')) + \
                   list(find_files(working_dir, '*log*'))

        for f in erasable:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)

    return c, subject_id, input_creds_path


def anat_workflow(sessions, conf, input_creds_path):
    # TODO ASH temporary code, remove
    # TODO ASH maybe scheme validation/normalization
    already_skullstripped = conf.already_skullstripped[0]
    if already_skullstripped == 2:
        already_skullstripped = 0
    elif already_skullstripped == 3:
        already_skullstripped = 1

    skullstrip_meth = {
        'anatomical_brain_mask': 'mask',
        'BET': 'fsl',
        'AFNI': 'afni'
    }



    subject_id = sessions[0]['subject_id']

    anat_preproc_list = []
    for ses in sessions:

        unique_id = ses['unique_id']
        if 'brain_mask' in ses.keys():
            if ses['brain_mask'] and ses[
                'brain_mask'].lower() != 'none':
                brain_flow = create_anat_datasource(
                    'brain_gather_%d' % unique_id)
                brain_flow.inputs.inputnode.subject = subject_id
                brain_flow.inputs.inputnode.anat = ses['brain_mask']
                brain_flow.inputs.inputnode.creds_path = input_creds_path
                brain_flow.inputs.inputnode.dl_dir = conf.workingDirectory

        if "AFNI" in conf.skullstrip_option:

        if "BET" in conf.skullstrip_option:

        wf = pe.Workflow(name='anat_preproc' + unique_id)
        anat_datasource = create_anat_datasource('anat_gather_%d' % unique_id)
        anat_datasource.inputs.inputnode.subject = subject_id
        anat_datasource.inputs.inputnode.anat = ses['anat']
        anat_datasource.inputs.inputnode.creds_path = input_creds_path
        anat_datasource.inputs.inputnode.dl_dir = conf.workingDirectory

        anat_prep = create_anat_preproc(skullstrip_meth[])
        anat_preproc_list.append(wf)

    template_creation_flirt([node.ouputs for node in anat_preproc_list])

    return


def anat_workflow_old(ses_list, conf, run, pipeline_timing_info=None,
                  p_name=None, plugin='MultiProc', plugin_args=None):
    """
    Function to prepare and, optionally, run the C-PAC workflow

    Parameters
    ----------
    sub_dict : dictionary
        subject dictionary with anatomical and functional image paths
    c : Configuration object
        CPAC pipeline configuration dictionary object
    run : boolean
        flag to indicate whether to run the prepared workflow
    pipeline_timing_info : list (optional); default=None
        list of pipeline info for reporting timing information
    p_name : string (optional); default=None
        name of pipeline
    plugin : string (optional); default='MultiProc'
        nipype plugin to utilize when the workflow is ran
    plugin_args : dictionary (optional); default=None
        plugin-specific arguments for the workflow plugin

    Returns
    -------
    workflow : nipype workflow
        the prepared nipype workflow object containing the parameters
        specified in the config
    """
    # TODO LONG_REG some modifications for the pipeline name for the logger
    c, subject_id, input_creds_path = init_subject_wf(ses_list[0], conf)

    # TODO LONG_REG
    # subject_info = {'subject_id': subject_id,
    #                 'start_time': c.pipeline_start_time
    #                 }

    # TODO ASH temporary code, remove
    # TODO ASH maybe scheme validation/normalization
    already_skullstripped = c.already_skullstripped[0]
    if already_skullstripped == 2:
        already_skullstripped = 0
    elif already_skullstripped == 3:
        already_skullstripped = 1

    """""""""""""""""""""""""""""""""""""""""""""""""""
     PREPROCESSING
    """""""""""""""""""""""""""""""""""""""""""""""""""

    strat_initial = Strategy()
    # The list of strategies that will be shared all along the pipeline creation
    strat_list = []

    num_strat = 0

    workflow_bit_id = {}
    workflow_counter = 0

    anat_flow = create_anat_datasource('anat_gather_%d' % num_strat)
    anat_flow.inputs.inputnode.subject = subject_id
    anat_flow.inputs.inputnode.anat = sub_dict['anat']
    anat_flow.inputs.inputnode.creds_path = input_creds_path
    anat_flow.inputs.inputnode.dl_dir = c.workingDirectory

    strat_initial.update_resource_pool({
        'anatomical': (anat_flow, 'outputspec.anat')
    })

    if 'brain_mask' in sub_dict.keys():
        if sub_dict['brain_mask'] and sub_dict['brain_mask'].lower() != 'none':
            brain_flow = create_anat_datasource('brain_gather_%d' % num_strat)
            brain_flow.inputs.inputnode.subject = subject_id
            brain_flow.inputs.inputnode.anat = sub_dict['brain_mask']
            brain_flow.inputs.inputnode.creds_path = input_creds_path
            brain_flow.inputs.inputnode.dl_dir = c.workingDirectory

            strat_initial.update_resource_pool({
                'anatomical_brain_mask': (brain_flow, 'outputspec.anat')
            })

    num_strat += 1
    strat_list.append(strat_initial)

    workflow_bit_id['anat_preproc'] = workflow_counter

    new_strat_list = []

    for num_strat, strat in enumerate(strat_list):

        if 'anatomical_brain_mask' in strat:
            anat_preproc = create_anat_preproc(method='mask',
                                               already_skullstripped=already_skullstripped,
                                               wf_name='anat_preproc_mask_%d' % num_strat)

            node, out_file = strat['anatomical_brain']
            workflow.connect(node, out_file, anat_preproc,
                             'inputspec.anat')

            node, out_file = strat['anatomical_brain_mask']
            workflow.connect(node, out_file,
                             anat_preproc, 'inputspec.brain_mask')

            strat.append_name(anat_preproc.name)
            strat.set_leaf_properties(anat_preproc, 'outputspec.brain')

            strat.update_resource_pool({
                'anatomical_brain': (anat_preproc, 'outputspec.brain'),
                'anatomical_reorient': (anat_preproc, 'outputspec.reorient'),
            })

            create_log_node(workflow, anat_preproc,
                            'outputspec.brain', num_strat)

    strat_list += new_strat_list

    new_strat_list = []

    for num_strat, strat in enumerate(strat_list):

        if 'anatomical_brain_mask' in strat:
            continue

        if "AFNI" not in c.skullstrip_option and "BET" not in c.skullstrip_option:
            err = '\n\n[!] C-PAC says: Your skull-stripping method options ' \
                  'setting does not include either \'AFNI\' or \'BET\'.\n\n' \
                  'Options you provided:\nskullstrip_option: {0}' \
                  '\n\n'.format(str(c.skullstrip_option))
            raise Exception(err)

        if "AFNI" in c.skullstrip_option:

            anat_preproc = create_anat_preproc(method='afni',
                                               already_skullstripped=already_skullstripped,
                                               wf_name='anat_preproc_afni_%d' % num_strat)

            anat_preproc.inputs.AFNI_options.set(
                shrink_factor=c.skullstrip_shrink_factor,
                var_shrink_fac=c.skullstrip_var_shrink_fac,
                shrink_fac_bot_lim=c.skullstrip_shrink_factor_bot_lim,
                avoid_vent=c.skullstrip_avoid_vent,
                niter=c.skullstrip_n_iterations,
                pushout=c.skullstrip_pushout,
                touchup=c.skullstrip_touchup,
                fill_hole=c.skullstrip_fill_hole,
                avoid_eyes=c.skullstrip_avoid_eyes,
                use_edge=c.skullstrip_use_edge,
                exp_frac=c.skullstrip_exp_frac,
                smooth_final=c.skullstrip_smooth_final,
                push_to_edge=c.skullstrip_push_to_edge,
                use_skull=c.skullstrip_use_skull,
                perc_int=c.skullstrip_perc_int,
                max_inter_iter=c.skullstrip_max_inter_iter,
                blur_fwhm=c.skullstrip_blur_fwhm,
                fac=c.skullstrip_fac,
            )

            node, out_file = strat['anatomical']
            workflow.connect(node, out_file,
                             anat_preproc, 'inputspec.anat')

            if "BET" in c.skullstrip_option:
                strat = strat.fork()
                new_strat_list.append(strat)

            strat.append_name(anat_preproc.name)
            strat.set_leaf_properties(anat_preproc, 'outputspec.brain')

            strat.update_resource_pool({
                'anatomical_brain': (anat_preproc, 'outputspec.brain'),
                'anatomical_reorient': (anat_preproc, 'outputspec.reorient'),
            })

            create_log_node(workflow, anat_preproc,
                            'outputspec.brain', num_strat)

    strat_list += new_strat_list

    new_strat_list = []

    for num_strat, strat in enumerate(strat_list):

        if 'anatomical_brain_mask' in strat:
            continue

        if 'anatomical_brain' in strat:
            continue

        if "BET" in c.skullstrip_option:
            anat_preproc = create_anat_preproc(method='fsl',
                                               already_skullstripped=already_skullstripped,
                                               wf_name='anat_preproc_bet_%d' % num_strat)

            anat_preproc.inputs.BET_options.set(
                frac=c.bet_frac,
                mask_boolean=c.bet_mask_boolean,
                mesh_boolean=c.bet_mesh_boolean,
                outline=c.bet_outline,
                padding=c.bet_padding,
                radius=c.bet_radius,
                reduce_bias=c.bet_reduce_bias,
                remove_eyes=c.bet_remove_eyes,
                robust=c.bet_robust,
                skull=c.bet_skull,
                surfaces=c.bet_surfaces,
                threshold=c.bet_threshold,
                vertical_gradient=c.bet_vertical_gradient,
            )

            node, out_file = strat['anatomical']
            workflow.connect(node, out_file, anat_preproc, 'inputspec.anat')

            strat.append_name(anat_preproc.name)
            strat.set_leaf_properties(anat_preproc, 'outputspec.brain')

            strat.update_resource_pool({
                'anatomical_brain': (anat_preproc, 'outputspec.brain'),
                'anatomical_reorient': (anat_preproc, 'outputspec.reorient'),
            })

            create_log_node(workflow, anat_preproc,
                            'outputspec.brain', num_strat)

    strat_list += new_strat_list