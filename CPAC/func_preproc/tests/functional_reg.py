
import os
import six
import ntpath
import numpy as np

import nibabel as nib
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype import MapNode


def simple_stats(image, method='mean', axis=3):
    """
    Replace zeros by ones and non-zero values by 1
    Parameters
    ----------
    image: str or nibabel.nifti1.Nifti1Image
        path to the nifti file to be averaged or
        the image already loaded through nibabel
    method: {'mean', 'median', 'std', 'average' ...}, optional
        method used to average the 4D image.
        most of the functions from numpy having only one array in mandatory
        parameter and an axis option will work.
        default is 'mean'
    axis: int, optional
        default is 3 (for the temporal resolution of fMRI for example)

    Returns
    -------
    output: nibabel.Nifti1Image

    Examples
    --------
    >>> fmri_img_path = 'sub1/func/sub1_task-rest_run-1_bold.nii.gz'
    >>> out_nii = simple_stats(fmri_img_path, method='median')

    """
    if isinstance(image, nib.nifti1.Nifti1Image):
        img = image
    elif isinstance(image, six.string_types):
        if not os.path.exists(image):
            raise ValueError(str(image) + " does not exist.")
        else:
            img = nib.load(image)
    else:
        raise TypeError("Image can be either a string or a nifti1.Nifti1Image")

    data = img.get_data()
    out_data = getattr(np, method)(data, axis)

    return nib.nifti1.Nifti1Image(out_data, img.affine)


def init_robust_template(wf_name='init_robust'):

    init_template = pe.Workflow(name=wf_name)

    inputspec = pe.Node(util.IdentityInterface(
        fields=['image',
                'method',
                'axis']),
        name='inputspec')

    outputspec = pe.Node(
        util.IdentityInterface(fields=['output_img']),
        name='outputspec')

    func_create_base = pe.Node(
        util.Function(input_names=['image',
                                   'method',
                                   'axis'],
                      output_names=['output_img'],
                      function=simple_stats,
                      as_module=True),
        name='func_create_base'
    )

    init_template.connect(inputspec, 'image',
                          func_create_base, 'image')
    init_template.connect(inputspec, 'method',
                          func_create_base, 'method')
    init_template.connect(inputspec, 'axis',
                          func_create_base, 'axis')

    init_template.connect(func_create_base, 'output_img',
                          outputspec, 'output_img')
    return init_template


def create_temporary_template(img_list, out_path, avg_method='median'):
    """
    Average all the 3D images of the list into one 3D image
    WARNING---the function assumes that all the images have the same header,
    the output image will have the same header as the first image of the list
    Parameters---
    ----------
    img_list: list of str
        list of images paths
    avg_method: str
        function names from numpy library such as 'median', 'mean', 'std' ...

    Returns
    -------
    tmp_template: Nifti1Image
    """
    if not img_list:
        print('ERROR create_temporary_template: image list is empty')

    if len(img_list) == 1:
        return img_list[0]

    avg_data = getattr(np, avg_method)(
        np.asarray([nib.load(img).get_data() for img in img_list]), 0)

    nii = nib.Nifti1Image(avg_data, nib.load(img_list[0]).affine)
    nib.save(nii, out_path)
    return out_path


def register_img_list(img_list, ref_img, output_folder, dof=12,
                      interp='trilinear', cost='corratio'):
    """
    Register a list of images to the reference image. The registered images are
    stored in output_folder.
    Parameters
    ----------
    img_list: list of str
        list of images paths
    ref_img: str
        path to the reference image to which the images will be registered
    output_folder: str
        path to the output directory
    dof: integer (int of long)
        number of transform degrees of freedom (FLIRT) (12 by default)
    interp: str
        ('trilinear' (default) or 'nearestneighbour' or 'sinc' or 'spline')
        final interpolation method used in reslicing
    cost: str
        ('mutualinfo' or 'corratio' (default) or 'normcorr' or 'normmi' or
         'leastsq' or 'labeldiff' or 'bbr')
        cost function

    Returns
    -------
    multiple_linear_reg: MapNode
        outputs.out_file will contain the registered images
        outputs.out_matrix_file will contain the transformation matrices
    """
    if not img_list:
        print('ERROR create_temporary_template: image list is empty')

    # output_folder = os.getcwd()

    output_img_list = [os.path.join(output_folder, ntpath.basename(img))
                       for img in img_list]

    output_mat_list = [os.path.join(output_folder,
                                    ntpath.basename(img).split('.')[0] + '.mat')
                       for img in img_list]

    linear_reg = fsl.FLIRT()

    multiple_linear_reg = MapNode(
        linear_reg,
        name="multiple_linear_reg",
        iterfield=["in_file",
                   "out_file",
                   "out_matrix_file"])

    # iterfields
    multiple_linear_reg.inputs.in_file = img_list
    multiple_linear_reg.inputs.out_file = output_img_list
    multiple_linear_reg.inputs.out_matrix_file = output_mat_list

    # fixed inputs
    multiple_linear_reg.inputs.cost = cost
    multiple_linear_reg.inputs.dof = dof
    multiple_linear_reg.inputs.interp = interp
    multiple_linear_reg.inputs.reference = ref_img

    multiple_linear_reg.outputs.out_file = output_img_list
    multiple_linear_reg.outputs.out_matrix_file = output_mat_list

    return multiple_linear_reg


def norm_transformation(flirt_mat):
    # Translation vector
    translation = flirt_mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_affine_transform = flirt_mat[0:3, 0:3]
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_affine_transform - np.identity(3), 'fro')
    return pow(tr_norm, 2) + pow(affine_norm, 2)


def template_convergence(mat_file,
                         convergence_threshold=np.finfo(np.float64).eps):
    mat1 = np.loadtxt(mat_file)
    distance = norm_transformation(mat1)

    return abs(distance) <= convergence_threshold


def template_creation_loop(img_list, output_folder,
                           init_reg=MapNode, avg_method='median', dof=12,
                           interp='trilinear', cost='corratio',
                           convergence_threshold=np.finfo(np.float64).eps):
    """

    Parameters
    ----------
    img_list: list of str
        list of images paths
    output_folder
    init_reg: nipype.MapNode
        the output of the function register_img_list with another reference
        Reuter et al. 2012 (NeuroImage) section "Improved template estimation"
        doi:10.1016/j.neuroimage.2012.02.084 recommend to use a ramdomly
        selected image from the input dataset
    avg_method: str
        function names from numpy library such as 'median', 'mean', 'std' ...
    dof: integer (int of long)
        number of transform degrees of freedom (FLIRT) (12 by default)
    interp: str
        ('trilinear' (default) or 'nearestneighbour' or 'sinc' or 'spline')
        final interpolation method used in reslicing
    cost: str
        ('mutualinfo' or 'corratio' (default) or 'normcorr' or 'normmi' or
         'leastsq' or 'labeldiff' or 'bbr')
        cost function
    convergence_threshold: float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.

    Returns
    -------
    template: str
        path to the final template

    """
    if not img_list:
        print('ERROR create_temporary_template: image list is empty')

    image_list = img_list
    converged = False
    tmp_template = os.path.join(output_folder, 'tmp_template.nii.gz')

    while not converged:
        tmp_template = create_temporary_template(image_list,
                                                 tmp_template,
                                                 avg_method='median')
        reg_list_node = register_img_list(image_list,
                                          tmp_template,
                                          output_folder,
                                          dof=12,
                                          interp='trilinear',
                                          cost='corratio')
        reg_list_node.run()

        image_list = reg_list_node.inputs.out_file
        mat_list = reg_list_node.inputs.out_matrix_file
        convergence_list = [template_convergence(
            mat, convergence_threshold) for mat in mat_list]
        converged = all(convergence_list)

    template = tmp_template
    return template


def mri_robust_template(wf_name='robust_template'):
    """

    Parameters
    ----------
    wf_name: str
        workflow's name

    Returns
    -------
    robust_template : workflow

    """

    robust_template = pe.Workflow(name=wf_name)

    inputspec = pe.Node(util.IdentityInterface(
        fields=['moving',
                'avg_method',
                'ref',
                'interp']),
        name='inputspec')

    outputspec = pe.Node(
        util.IdentityInterface(fields=['transform_linear_xfm']),
        name='outputspec')

    init_wf = init_robust_template()

    linear_reg = pe.Node(interface=fsl.FLIRT(),
                         name='linear_func_to_anat')
    linear_reg.inputs.cost = 'corratio'
    linear_reg.inputs.dof = 12
    linear_reg.inputs.interp = 'trilinear'

    # Initialization of the within subject template creation
    robust_template.connect(inputspec, 'moving',
                            init_wf, 'image')

    robust_template.connect(inputspec, 'ref',
                            init_wf, 'image')

    robust_template.inputs.inputspec.axis = 4

    robust_template.connect(init_wf, 'output_img',
                            linear_reg, 'in_file')

    robust_template.connect(inputspec, 'reference_brain',
                            linear_reg, 'reference')

    # robust_template.connect(inputspec, 'interp',
    #                         linear_reg, 'interp')

    robust_template.connect(linear_reg, 'out_matrix_file',
                            outputspec, 'transform_linear_xfm')

    return robust_template
