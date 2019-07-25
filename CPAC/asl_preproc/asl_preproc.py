#!/usr/bin/env python

# PIPELINE FOR ASL ANALYSIS #

# -----------------------------------------------------------------------------------------------------#
# Import modules
# -----------------------------------------------------------------------------------------------------#
from nipype import logging
logger = logging.getLogger('workflow')
import os, subprocess
from os.path import join as opj
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
from nipype.interfaces.afni import preprocess

from CPAC.warp.pipeline import ants_apply_warps_asl_mni
from CPAC.registration import (
    create_bbregister_asl_to_anat,
    create_register_asl_to_anat
)


def split_pairs(asl_data, iaf = 'tc'):
    """
    use asl_file to extract label and control images, for moco separately

    Example in bash:
    >>> asl_file --data=/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/func/sub-A00073953_ses-ALGA_task-rest_pcasl.nii.gz --ntis=1 --iaf=$iaf --spairs --out=/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/func/asl_file_test_local

    outputs: '_even.nii.gz' and '_odd.nii.gz'

    Example:

    >>> iaf='tc'
    >>> split_pairs_imports = ['import os', 'import subprocess']
    >>> split_ASL_pairs = pe.Node(interface=util.Function(input_names=['in_file',
    >>>                                                     'iaf',
    >>>                                                     'output_dir'],
    >>>                                     output_names=['control_image',
    >>>                                                     'label_image'],
    >>>                                     function=split_pairs,
    >>>                                     imports=split_pairs_imports),
    >>>             name='split_pairs')
    >>> split_ASL_pairs.interface.num_threads = num_threads
    """
    output_dir = os.getcwd()

    splitcmd = ["asl_file",
                "--data=" + asl_data,
                "--ntis=" + "1",
                "--iaf=" + 'tc',
                "--spairs",
                "--out=" + output_dir + '/asldata' ]
#  
    # write out the actual command-line entry for testing/validation later
    command_file = os.path.join(os.getcwd(), 'command.txt')
    with open(command_file, 'wt') as f:
        f.write(' '.join(splitcmd))
    try:
        retcode = subprocess.check_output(splitcmd)
    except Exception as e:
        raise Exception('[!] asl_file did not complete successfully.'
                        '\n\nError details:\n{0}\n{1}\n'.format(e, e.output))
 
    # change suffix naming from _even _odd to _label and _control,
    if iaf is 'tc':
        suffix_even = 'label'
        suffix_odd = 'control'
    elif iaf is 'ct':
        suffix_even = 'control'
        suffix_odd = 'label'
  
    # rename output files from asl_file
    os.rename(output_dir + '/asldata_even.nii.gz', output_dir + 'asldata_' + suffix_even + '.nii.gz')
    os.rename(output_dir + '/asldata_odd.nii.gz', output_dir + 'asldata_' + suffix_odd + '.nii.gz')

    control_image = output_dir + 'asldata_control.nii.gz'
    label_image = output_dir + 'asldata_label.nii.gz'

    return control_image, label_image


def diffdata(asl_file, iaf = 'tc'):
    """
    runs the FSL tool asl_file to get subtracted image (tag minus control)
    Example:

    >>> iaf='tc'
    >>> diffdata_imports = ['import os', 'import subprocess']
    >>> run_diffdata = pe.Node(interface=util.Function(input_names=['in_file',
    >>>                                                     'iaf',
    >>>                                                     'output_dir'],
    >>>                                     output_names=['diffdata_image',
    >>>                                                   'diffdata_mean'],
    >>>                                     function=diffdata,
    >>>                                     imports=diffdata_imports),
    >>>             name='diffdata')

    """
    output_dir = os.getcwd()

    diffdatacmd = ["asl_file",
                   "--data=" + asl_file,
                   "--ntis=" + "1",
                   "--iaf=" + 'tc',
                   "--diff",
                   "--mean=" + output_dir + '/diffdata_mean',
                   "--out=" + output_dir + '/diffdata']
    #
    # write out the actual command-line entry for testing/validation later
    command_file = os.path.join(os.getcwd(), 'command.txt')
    with open(command_file, 'wt') as f:
        f.write(' '.join(diffdatacmd))
    try:
        retcode = subprocess.check_output(diffdatacmd)
    except Exception as e:
        raise Exception('[!] asl_file did not complete successfully.'
                        '\n\nError details:\n{0}\n{1}\n'.format(e, e.output))

    diffdata_image = None
    diffdata_mean = None

    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    for f in files:
        if "diffdata.nii.gz" in f:
            diffdata_image = os.getcwd() + "/" + f
        elif "diffdata_mean.nii.gz" in f:
            diffdata_mean = os.getcwd() + "/" + f

    if not diffdata_image:
        raise Exception("\n\n[!] No diffdata output file found. "
                        "asl_file may not have completed "
                        "successfully.\n\n")
    if not diffdata_mean:
        raise Exception("\n\n[!] No diffdata_mean output file found. "
                        "asl_file may not have completed "
                        "successfully.\n\n")

    return diffdata_image, diffdata_mean


def oxford_asl(asl_file, anatomical_skull, anatomical_brain, seg):
    """
    Call oxford_asl for asl preprocessing and cbf calc
    Default to use asl_reg for registration. Can use ANTs if flag supplied

        >>> asl_file='/Users/tbw665/Documents/data/sourcedata/sub-A00073953/ses-ALGA/func/sub-A00073953_ses-ALGA_task-rest_asl.nii.gz'
        >>> t1='/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/anat/sub-A00073953_ses-ALGA_T1w.anat/T1.nii.gz'
        >>> t1_brain='/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/anat/sub-A00073953_ses-ALGA_T1w.anat/T1_biascorr_brain.nii.gz'
        >>> iaf='tc'
        >>> t1_segdir='/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/anat/sub-A00073953_ses-ALGA_T1w.anat/T1_fast'
        >>> oxford_asl(in_file, output_dir, t1, t1_brain, t1_segdir, iaf)

    outputs:
    perfusion.nii.gz: mean perfusion image, provides blood flow in relative (scanner) units
    diffdata.nii.gz = subtracted image (tag minus control)

    """
    output_dir = str(os.getcwd())

    # get path and image stub of FAST pve output
    fastsrc = str(seg.split('_pve_2.nii.gz')[0])

    print '********oxford_asl: ' + output_dir + ' ' + fastsrc + ' **********'
    #/outputs/working/resting_preproc_sub-A00073953_ses-ALGA/seg_preproc_0/segment/segment
    aslcmd = ["oxford_asl",
              "-i", asl_file,
              "-o", output_dir,
              "--spatial",
              "--mc",
              "--wp",
              "--tis=2.5",
              "--bolus=1.5",
              "--iaf=tc",
              "--casl",
              "-s", anatomical_skull,
              "--sbrain=" + anatomical_brain,
              "--fastsrc=/outputs/working/resting_preproc_sub-A00073953_ses-ALGA/seg_preproc_0/segment/segment",
              "--verbose=1",
              "--debug"]

    # write out the actual command-line entry for testing/validation later
    command_file = os.path.join(os.getcwd(), 'command.txt')
    with open(command_file, 'wt') as f:
        f.write(' '.join(aslcmd))

    try:
        retcode = subprocess.check_output(aslcmd)
    except Exception as e:
        raise Exception('[!] oxford_asl did not complete successfully.' + str(e))



    perfusion_image = None

    files = [f for f in os.listdir('./native_space') if os.path.isfile('./native_space/' + f)]

    for f in files:
        if "perfusion" in f:
            perfusion_image = os.getcwd() + "/" + f

    if not perfusion_image:
        raise Exception("\n\n[!] No perfusion output file found. "
                        "oxford_asl may not have completed "
                        "successfully.\n\n")

    asl2anat = None

    # files = [f for f in os.listdir('./native_space') if os.path.isfile('./native_space/' + f)]
    #
    # for f in files:
    #     if "asl2struct" in f:
    #         asl2anat = os.getcwd() + "/" + f

    for r, d, f in os.walk(output_dir):
        for file in f:
            if 'asl2struct.nii.gz' in file:
                asl2anat = os.path.join(r, file)

    print asl2anat

    if not asl2anat:
        raise Exception("\n\n[!] No asl2anat output file found. "
                        "oxford_asl may not have completed "
                        "successfully.\n\n")

    asl2anat_linear_xfm = None

    files = [f for f in os.listdir('./native_space') if os.path.isfile('./native_space/' + f)]

    for f in files:
        if "asl2struct.mat" in f:
            asl2anat_linear_xfm = os.getcwd() + "/" + f

    print asl2anat_linear_xfm

    if not asl2anat_linear_xfm:
        raise Exception("\n\n[!] No asl2anat_linear_xfm output file found. "
                        "oxford_asl may not have completed "
                        "successfully.\n\n")

    return perfusion_image, asl2anat, asl2anat_linear_xfm


def create_asl_preproc(c, strat, wf_name='asl_preproc'):
    # resource_pool = strat?
    # print('resource pool asl preproc: ', str(strat.get_resource_pool()))

    # allocate a workflow object
    asl_workflow = pe.Workflow(name=wf_name)
    asl_workflow.base_dir = c.workingDirectory


    # configure the workflow's input spec
    inputNode = pe.Node(util.IdentityInterface(fields=['asl_file',
                                                       'anatomical_skull',
                                                       'anatomical_brain',
                                                       'seg_wm_pve']),
                        name='inputspec')

    # configure the workflow's output spec
    outputNode = pe.Node(util.IdentityInterface(fields=['meanasl',
                                                        'perfusion_image',
                                                        'diffdata',
                                                        'diffdata_mean']),

                         name='outputspec')

    # get segmentation output dir and file stub


    # create nodes for de-obliquing and reorienting
    try:
        from nipype.interfaces.afni import utils as afni_utils
        func_deoblique = pe.Node(interface=afni_utils.Refit(),
                                 name='func_deoblique')
    except ImportError:
        func_deoblique = pe.Node(interface=preprocess.Refit(),
                                 name='func_deoblique')
    func_deoblique.inputs.deoblique = True

    asl_workflow.connect(inputNode, 'asl_file',
                         func_deoblique, 'in_file')

    try:
        func_reorient = pe.Node(interface=afni_utils.Resample(),
                                name='func_reorient')
    except UnboundLocalError:
        func_reorient = pe.Node(interface=preprocess.Resample(),
                                name='func_reorient')

    func_reorient.inputs.orientation = 'RPI'
    func_reorient.inputs.outputtype = 'NIFTI_GZ'

    # connect deoblique to reorient
    asl_workflow.connect(func_deoblique, 'out_file',
                         func_reorient, 'in_file')

    # create node for splitting control and label pairs (unused currently)
    split_pairs_imports = ['import os', 'import subprocess']
    split_ASL_pairs = pe.Node(interface=util.Function(input_names=['asl_file'],
                                                      output_names = ['control_image',
                                                                      'label_image'],
                                                      function = split_pairs,
                                                      imports = split_pairs_imports),
                              name = 'split_pairs')

    # create node for calculating subtracted images
    diffdata_imports = ['import os', 'import subprocess']
    run_diffdata = pe.Node(interface=util.Function(input_names=['asl_file'],
                                                   output_names = ['diffdata_image',
                                                                   'diffdata_mean'],
                                                   function = diffdata,
                                                   imports = diffdata_imports),
                           name = 'diffdata')

    asl_workflow.connect(func_reorient, 'out_file',
                         run_diffdata, 'asl_file')

    asl_workflow.connect(run_diffdata, 'diffdata_image',
                         outputNode, 'diffdata')

    asl_workflow.connect(run_diffdata, 'diffdata_mean',
                         outputNode, 'diffdata_mean')

    # create node for oxford_asl (perfusion image)

    asl_imports = ['import os', 'import subprocess']
    run_oxford_asl = pe.Node(interface=util.Function(input_names=['asl_file',
                                                                  'anatomical_skull',
                                                                  'anatomical_brain',
                                                                  'seg'],
                                                     output_names=['perfusion_image',
                                                                   'asl2anat_linear_xfm',
                                                                   'asl2anat'],
                                                     function=oxford_asl,
                                                     imports=asl_imports),
                             name='run_oxford_asl')

    # wire inputs from resource pool to ASL preprocessing FSL script

    # connect output of reorient to run_oxford_asl
    asl_workflow.connect(func_reorient, 'out_file',
                         run_oxford_asl, 'asl_file')

    asl_workflow.connect(inputNode, 'seg_wm_pve',
                         run_oxford_asl, 'seg')

    # pass the anatomical to the workflow
    asl_workflow.connect(inputNode, 'anatomical_skull',
                         run_oxford_asl,
                         'anatomical_skull')

    # pass the anatomical to the workflow
    asl_workflow.connect(inputNode, 'anatomical_brain',
                         run_oxford_asl, 'anatomical_brain')

    # connect oxford_asl outputs to outputNode

    asl_workflow.connect(run_oxford_asl, 'asl2anat_linear_xfm',
                         outputNode, 'asl2anat_linear_xfm')

    asl_workflow.connect(run_oxford_asl, 'asl2anat',
                         outputNode, 'asl2anat')

    asl_workflow.connect(run_oxford_asl, 'perfusion_image',
                         outputNode, 'perfusion_image')

    strat.update_resource_pool({
        'mean_asl_in_anat': (run_oxford_asl, 'anat_asl'),
        'asl_to_anat_linear_xfm': (run_oxford_asl, 'asl2anat_linear_xfm')
    })

    # Take mean of the asl data for registration

    try:
        get_mean_asl = pe.Node(interface=afni_utils.TStat(),
                               name='get_mean_asl')
    except UnboundLocalError:
        get_mean_asl = pe.Node(interface=preprocess.TStat(),
                               name='get_mean_asl')

    get_mean_asl.inputs.options = '-mean'
    get_mean_asl.inputs.outputtype = 'NIFTI_GZ'

    asl_workflow.connect(func_reorient, 'out_file',
                         get_mean_asl, 'in_file')

    asl_workflow.connect(get_mean_asl, 'out_file',
                         outputNode, 'meanasl')

    return asl_workflow
