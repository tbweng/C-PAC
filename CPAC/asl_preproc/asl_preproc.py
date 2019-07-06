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

""" 
# load data and init variables
from nibabel import load

in_file = '/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/func/sub-A00073953_ses-ALGA_task-rest_pcasl.nii.gz'
img = load(in_file)
img.data = img.get_data()
img.hdr = img.header
img.dim = img.shape
img.fname = in_file
img.path = '/'.join(in_file.split('/')[:-1]) + '/'
img.file = in_file.split('/')[-1].split('.')[0]
output_dir = '/'.join(in_file.split('/')[:-1]) + '/'

# -----------------------------------------------------------------------------------------------------#
# Specify variables
# -----------------------------------------------------------------------------------------------------#

# Defining list of subjects you want to run the pipeline on.
# This assumes that your functional and structural scans are in the same folder
# named subject_id

subject_list = ['A00073953']
session_list = ['ALGA']

# Location of the experiment folder

experiment_dir = '/Users/tbw665/Documents/data/sourcedata'
output_dir = opj(experiment_dir, 'derivatives/')
working_dir = opj(experiment_dir, 'workingDir/')

# -----------------------------------------------------------------------------------------------------#
# Specify nodes
# -----------------------------------------------------------------------------------------------------#
# load data
# func_deoblique
# func_reorient
"""


# use asl_file to extract label and control images, for moco separately
"""
Example in bash:
>>> asl_file --data=/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/func/sub-A00073953_ses-ALGA_task-rest_pcasl.nii.gz --ntis=1 --iaf=$iaf --spairs --out=/Users/tbw665/Documents/data/sub-A00073953/ses-ALGA/func/asl_file_test_local

outputs: '_even.nii.gz' and '_odd.nii.gz'

"""

def split_pairs(asl_data, output_dir, iaf='tc'):
    """
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

    splitcmd = ["asl_file",
                "--data=" + asl_data,
                "--ntis=" + "1",
                "--iaf=" + iaf,
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

def diffdata(asl_data, output_dir, iaf='tc'):
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

    diffdatacmd = ["asl_file",
                "--data=" + asl_data,
                "--ntis=" + "1",
                "--iaf=" + iaf,
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

# asl_motion_correct (option: before or after asl_file)

#
# asl_motion_correct = pe.Node(interface=preprocess.Volreg(),
#                               name='asl_motion_correct')
# asl_motion_correct.inputs.args = '-Fourier -twopass'
# asl_motion_correct.inputs.zpad = 4
# asl_motion_correct.inputs.outputtype = 'NIFTI_GZ'


# if label and control images are motion corrected separately, need to merge back into 4D for subtraction?

# func_get_brain_mask
#
# func_get_brain_mask = pe.Node(interface=preprocess.Automask(),
#                               name='func_get_brain_mask')
#
# func_get_brain_mask.inputs.outputtype = 'NIFTI_GZ'

# func_spatial_smooth

# func_spatial_smooth = pe.Node(interface=preprocess.BlurToFWHM(),
#                               name='spatial_smooth')
# func_spatial_smooth.inputs.in_file = in_file
# func_spatial_smooth.inputs.fwhm = 5

# func_int_norm (intensity normalization by mean)
# Normalise() {
#     if [ ! -z $pvexist ]; then
# 	# also output the perfusion images normalised by the mean mask value - the parameter must have been output first for this to work
# 	#NB we only do this is the PVE are available (and thus the reqd masks will exist)
# 	parname=$1
# 	subdir=$2
# 	masktype=$3
#
# 	# get normalization from reported value in the output directory
# 	normval=`cat $outdir/native_space/$subdir/${parname}_${masktype}_mean.txt`
#
# 	if [ ! -z $stdout ]; then
# 	    fslmaths $outdir/std_space/$subdir/$parname -div $normval $outdir/std_space/$subdir/${parname}_norm
# 	fi
# 	if [ ! -z $nativeout ]; then
# 	    fslmaths $outdir/native_space/$subdir/$parname -div $normval $outdir/native_space/$subdir/${parname}_norm
# 	fi
# 	if [ ! -z $structout ]; then
# 	    fslmaths $outdir/struct_space/$subdir/$parname -div $normval $outdir/struct_space/$subdir/${parname}_norm
# 	fi
#
#     fi
# }

# cpac int norm
# int_norm = pe.Node(interface=fsl.ImageMaths(),
#                              name='int_norm')
# int_norm.inputs.op_string = '-ing 10000'
# int_norm.inputs.out_data_type = 'float'


# cbf_calc
# asl_file --data=$tempdir/asldata --ntis=$ntis --ibf=$ibf --iaf=$iaf --obf=tis --split --out=$tempdir/asldata --mean=$tempdir/asldata_mean

# USE asl_reg to transform ASL Scan to T1 space
# Registration() {
#     echo "Performing registration"
#     regbase=$1 #the i/p to the function is the image to use for registration
#     transopt=$2 # other options to pass to asl_reg
#     distout=$3 # we want to do distortion correction and save in the subdir distout
#
#     extraoptions=" "
#     if [ ! -z $lowstrucflag ]; then
# 	extraoptions=$extraoptions"-r $tempdir/lowstruc"
#     fi
#     if [ ! -z $debug ]; then
# 	extraoptions=$extraoptions" --debug"
#     fi
#
#     #if [ ! -z $reginit ]; then
#     #    extraoptions=$extraoptions" -c $reginit"
#     #fi
#
#     if [ -z $distout ]; then
# 	# normal registration
# 	$asl_reg -i $regbase -o $tempdir -s $tempdir/struc --sbet $tempdir/struc_bet $transopt $extraoptions
#
# 	if [ ! -z $trans ]; then
# 	    # compute the transformation needed to standard space if we have the relvant structural to standard transform
# 	    convert_xfm -omat $tempdir/asl2std.mat -concat $trans $tempdir/asl2struc.mat
# 	fi
#
#     else
# 	# registration for distortion correction
# 	fmapregstr=""
# 	if [ ! -z $nofmapreg ]; then
# 	    fmapregstr="--nofmapreg"
# 	fi
# 	$asl_reg -i $regbase -o $tempdir/$distout -s $tempdir/struc --sbet $tempdir/struc_bet $transopt $extraoptions --fmap=$tempdir/fmap --fmapmag=$tempdir/fmapmag --fmapmagbrain=$tempdir/fmapmagbrain --pedir=$pedir --echospacing=$echospacing $fmapregstr
#     fi
#
#
# }


def oxford_asl(asl_data, output_dir, t1, t1_brain, iaf='tc'):
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

    aslcmd = ["oxford_asl",
              "-i", asl_data,
              "-o", output_dir,
              "--spatial",
              "--mc",
              "--wp",
              "--tis", 2.5,
              "--bolus", 1.5,
              "--iaf=" + iaf,
              "--casl",
              "-s=" + t1,
              "--sbrain=" + t1_brain,
              "--fastsrc=" + 'fast_']


    try:
        retcode = subprocess.check_output(aslcmd)
    except Exception as e:
        raise Exception('[!] oxford_asl did not complete successfully.'
                        '\n\nError details:\n{0}\n{1}\n'.format(e, e.output))

    # write out the actual command-line entry for testing/validation later
    command_file = os.path.join(os.getcwd(), 'command.txt')
    with open(command_file, 'wt') as f:
        f.write(' '.join(aslcmd))

    perfusion_image = None

    files = [f for f in os.listdir('./native_space') if os.path.isfile('./native_space/' + f)]

    for f in files:
        if "perfusion" in f:
            perfusion_image = os.getcwd() + "/" + f

    if not perfusion_image:
        raise Exception("\n\n[!] No perfusion output file found. "
                        "oxford_asl may not have completed "
                        "successfully.\n\n")

    perfusion_image_mni = None

    files = [f for f in os.listdir('./std_space') if os.path.isfile('./std_space/' + f)]

    for f in files:
        if "perfusion" in f:
            perfusion_image_mni = os.getcwd() + "/" + f

    if not perfusion_image:
        raise Exception("\n\n[!] No perfusion_mni output file found. "
                        "oxford_asl may not have completed "
                        "successfully.\n\n")

    return perfusion_image, perfusion_image_mni


def pick_wm(seg_prob_list):
    seg_prob_list.sort()
    return seg_prob_list[-1]


def create_asl_preproc(asl_paths_dict, subject_id, c, strat, num_strat, num_ants_cores, wf_name='asl_preproc'):
    # resource_pool = strat?

   # allocate a workflow object
    workflow = pe.Workflow(name=wf_name)

    # configure the workflow's input spec
    inputNode = pe.Node(util.IdentityInterface(fields=['asl_file']),
                        name='inputspec')

    # configure the workflow's output spec
    outputNode = pe.Node(util.IdentityInterface(fields=['meanasl',
                                                        'perfusion_native',
                                                        'diffdata_native',
                                                        'diffdata_mean_native',
                                                        'perfusion_mni',
                                                        'diffdata_mni',
                                                        'diffdata_mean_mni']),

                         name='outputspec')
    # create data source and pass in data_config path information
    from CPAC.utils.datasource import create_func_datasource

    datasource_wf = create_func_datasource(asl_paths_dict,
                                     'func_gather_%d' % num_strat)

    datasource_wf.inputs.inputnode.set(
        subject=subject_id,
        dl_dir=c.workingDirectory
    )
    datasource_wf.get_node('inputnode').iterables = \
        ("scan", asl_paths_dict.keys())



    # print(data_config)


    # create nodes for de-obliquing and reorienting
    try:
        from nipype.interfaces.afni import utils as afni_utils
        func_deoblique = pe.Node(interface=afni_utils.Refit(),
                                 name='func_deoblique')
    except ImportError:
        func_deoblique = pe.Node(interface=preprocess.Refit(),
                                 name='func_deoblique')
    func_deoblique.inputs.deoblique = True

    workflow.connect(inputNode, 'asl_file',
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
    workflow.connect(func_deoblique, 'out_file',
                     func_reorient, 'in_file')




    # create node for splitting control and label pairs
    split_pairs_imports = ['import os', 'import subprocess']
    split_ASL_pairs = pe.Node(interface=util.Function(input_names=['in_file',
                                                                   'iaf',
                                                                   'output_dir'],
                                                      output_names = ['control_image',
                                                                      'label_image'],
                                                      function = split_pairs,
                                                      imports = split_pairs_imports),
                              name = 'split_pairs')


    # create node for calculating subtracted images
    diffdata_imports = ['import os', 'import subprocess']
    run_diffdata = pe.Node(interface=util.Function(input_names=['asl_file',
                                                                'iaf',
                                                                'output_dir'],
                                                   output_names = ['diffdata_image',
                                                                   'diffdata_mean'],
                                                   function = diffdata,
                                                   imports = diffdata_imports),
                           name = 'diffdata')

    # create node for oxford_asl (perfusion image)

    asl_imports = ['import os', 'import subprocess']
    run_oxford_asl = pe.Node(interface=util.Function(input_names=['asl_file',
                                                                  'output_dir',
                                                                  'iaf',
                                                                  'anatomical_skull',
                                                                  'anatomical_brain',
                                                                  'fast_out_prefix'],
                                                     output_names=['perfusion_image'],
                                                     function=oxford_asl,
                                                     imports=asl_imports),
                             name='run_oxford_asl')

    # wire inputs from resource pool to ASL preprocessing FSL script

    # connect output of reorient to run_oxford_asl
    workflow.connect(func_reorient, 'out_file',
                     run_oxford_asl, 'asl_file')

    workflow.connect(func_reorient, 'out_file',
                     run_diffdata, 'asl_file')

    # get the reorient skull-on anatomical from resource
    # pool
    node, out_file = strat['anatomical_reorient']

    # pass the anatomical to the workflow
    workflow.connect(node, out_file,
                     run_oxford_asl,
                     'anatomical_skull')

    # get the skullstripped anatomical from resource pool
    node, out_file = strat['anatomical_brain']

    # pass the anatomical to the workflow
    workflow.connect(node, out_file,
                     run_oxford_asl,
                     'anatomical_brain')



    # create nodes to write subtracted time series into MNI
    # asl -> T1, T1 -> MNI (needs to be done before asl preproc)
    # create_bbregister_func_to_anat (asl -> T1), then apply_ants_warp (T1 -> MNI), then ants_apply_warps_func_mni (asl -> MNI)

    #################
    ### asl -> T1 ###
    #################

    # Take mean of the asl data for registration

    try:
        get_mean_asl = pe.Node(interface=afni_utils.TStat(),
                               name='get_mean_asl')
    except UnboundLocalError:
        get_mean_asl = pe.Node(interface=preprocess.TStat(),
                               name='get_mean_asl')

    get_mean_asl.inputs.options = '-mean'
    get_mean_asl.inputs.outputtype = 'NIFTI_GZ'

    workflow.connect(func_reorient, 'out_file',
                     get_mean_asl, 'in_file')

    # TODO: needed?
    workflow.connect(get_mean_asl, 'out_file',
                     outputNode, 'meanasl')

    # Func -> T1 Registration (Initial Linear reg)

    dist_corr=False
    asl_to_anat = create_register_asl_to_anat(dist_corr,
                                                'asl_to_anat_FLIRT'
                                                '_%d' % num_strat)

    # Input registration parameters
    asl_to_anat.inputs.inputspec.interp = 'trilinear'

    # Input functional image (mean functional)
    workflow.connect(get_mean_asl, 'out_file',
                     asl_to_anat, 'inputspec.asl')

    # Input skull-stripped anatomical (anat.nii.gz)
    node, out_file = strat['anatomical_brain']
    workflow.connect(node, out_file,
                     asl_to_anat, 'inputspec.anat')

    strat.update_resource_pool({
        'mean_asl_in_anat': (asl_to_anat, 'outputspec.anat_asl_nobbreg'),
        'asl_to_anat_linear_xfm': (asl_to_anat, 'outputspec.asl_to_anat_linear_xfm_nobbreg')
    })

    # Func -> T1 Registration (BBREG)

    asl_to_anat_bbreg = create_bbregister_asl_to_anat(
        dist_corr,
        'func_to_anat_bbreg_%d' % num_strat
    )

    # Input registration parameters
    # TODO: $FSLDIR
    asl_to_anat_bbreg.inputs.inputspec.bbr_schedule = \
        '/usr/share/fsl/6.0/etc/flirtsch/bbr.sch'

    # Input mean asl image
    workflow.connect(get_mean_asl, 'out_file',
                     asl_to_anat_bbreg, 'inputspec.asl')


    # Input anatomical whole-head image (reoriented)
    node, out_file = strat['anatomical_reorient']
    workflow.connect(node, out_file,
                     asl_to_anat_bbreg,
                     'inputspec.anat_skull')

    # Input initial linear reg
    workflow.connect(asl_to_anat, 'outputspec.asl_to_anat_linear_xfm_nobbreg',
                     asl_to_anat_bbreg,
                     'inputspec.linear_reg_matrix')

    # Input segmentation probability maps for white matter
    # segmentation

    # node, out_file = strat['seg_probability_maps']
    # workflow.connect(node, (out_file, pick_wm),
    #                  asl_to_anat_bbreg,
    #                  'inputspec.anat_wm_segmentation')


    # T1 -> MNI done in main pipeline


    ##################
    ### asl -> MNI ###
    ##################

    warp_diffdata_wf = ants_apply_warps_asl_mni(
        workflow, strat, num_strat, num_ants_cores,
        run_diffdata, 'diffdata_image',
        get_mean_asl, 'out_file',
        c.template_brain_only_for_func,
        "diffdata_to_standard",
        "Linear", 3
    )

    # create_log_node(workflow, warp_mean_wf,
    #                'outputspec.output_image', num_strat)

    warp_perfusion_wf = ants_apply_warps_asl_mni(
        workflow, strat, num_strat, num_ants_cores,
        run_oxford_asl, 'perfusion_image',
        get_mean_asl, 'out_file',
        c.template_brain_only_for_func,
        "perfusion_image_to_standard",
        "Linear", 3
    )

    # create_log_node(workflow, warp_mean_wf,
    #               'outputspec.output_image', num_strat)


    # outputs to resourcepool, then to data sink in cpac_pipeline.py
    # TODO: need to both resource pool and data sink? Does resource pool get datasinked in cpac_pipeline?

    strat.update_resource_pool({
        'perfusion_native': (run_oxford_asl, 'outputspec.perfusion_image'),
        'diffdata_native': (run_diffdata, 'outputspec.diffdata'),
        'diffdata_mean_native': (run_diffdata, 'outputspec.diffdata_mean'),
        'diffdata_mni': (warp_diffdata_wf, 'outputspec.diffdata_to_standard'),
        'perfusion_mni': (warp_perfusion_wf, 'outputspec.perfusion_image_to_standard'),
    })


    # create data sinks for subtracted time series, in and out of MNI

    ds = pe.Node(nio.DataSink(), name='asl_sink')
    ds.inputs.base_directory = c.outputDirectory

    workflow.connect(run_oxford_asl, 'outputspec.perfusion_image',
               ds, 'asl.perfusion_image')
    workflow.connect(warp_perfusion_wf, 'outputspec.output_image', ds, 'asl.perfusion_mni')
    workflow.connect(run_diffdata, 'outputspec.diffdata',
               ds, 'asl.diffdata')
    workflow.connect(run_diffdata, 'outputspec.diffdata_mean', ds, 'asl.diffdata_mean')
    workflow.connect(warp_diffdata_wf, 'outputspec.output_image', ds, 'asl.diffdata_mni')

    # workflow.write_graph(graph2use='flat')
    # workflow.write_graph(dotfilename='./graph.dot', graph2use='hierarchical')
    # workflow.run('MultiProc', plugin_args={'n_procs': 2})
    return workflow
