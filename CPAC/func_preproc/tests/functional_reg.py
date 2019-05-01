
import os
import ntpath
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


import nibabel as nib
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype import MapNode
from CPAC.utils.nifti_utils import nifti_image_input


def read_ants_mat(ants_mat_file):
    if not os.path.exists(ants_mat_file):
        raise ValueError(str(ants_mat_file) + " does not exist.")

    with open(ants_mat_file) as f:
        for line in f:
            tmp = line.split(':')
            if tmp[0] == 'Parameters':
                oth_transform = np.reshape(
                    np.fromstring(tmp[1], float, sep=' '), (-1, 3))
            if tmp[0] == 'FixedParameters':
                translation = np.fromstring(tmp[1], float, sep=' ')
    return translation, oth_transform


def read_flirt_mat(flirt_mat):
    if isinstance(flirt_mat, np.ndarray):
        mat = flirt_mat
    elif isinstance(flirt_mat, str):
        if os.path.exists(flirt_mat):
            mat = np.loadtxt(flirt_mat)
        else:
            raise IOError("ERROR norm_transformation: " + flirt_mat +
                          " file does not exist")
    else:
        raise TypeError("ERROR norm_transformation: flirt_mat should be" +
                        " either a str or a numpy.ndarray matrix")

    if mat.shape != (4, 4):
        raise ValueError("ERROR norm_transformation: the matrix should be 4x4")

    # Translation vector
    translation = mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_transform = mat[0:3, 0:3]

    return translation, oth_transform


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
    img = nifti_image_input(image)

    data = img.get_data()
    out_data = getattr(np, method)(data, axis)

    return nib.nifti1.Nifti1Image(out_data, img.affine)


def average_images(images_list, method='median', axis=3):
    """
    Average each image of images_list with the given method and in the given
    axis into a list of lower dimension images. (Typically 4D to 3D)
    Parameters
    ----------
    images_list: list of str
        list of images (typically 4D functional images) paths
    method: {'mean', 'median', 'std', 'average' ...}, optional
        method used to average the 4D image.
        most of the functions from numpy having only one array in mandatory
        parameter and an axis option will work.
        default is 'mean'
    axis: int, optional
        default is 3 (for the temporal resolution of fMRI for example)

    Returns
    -------

    """
    img_list = [simple_stats(
        img, method=method, axis=axis) for img in images_list]
    return img_list


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
        raise ValueError('ERROR create_temporary_template: image list is empty')

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
        raise ValueError('ERROR register_img_list: image list is empty')

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


def norm_transformations(translation, oth_transform):
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_transform - np.identity(3), 'fro')
    return pow(tr_norm, 2) + pow(affine_norm, 2)


def norm_transformation(flirt_mat):
    """
    Calculate the squared norm of the translation + squared Frobenium norm
    of the difference between other affine transformations and the identity
    from an fsl FLIRT transformation matrix
    Parameters
    ----------
    flirt_mat: str or numpy.ndarray
        Either the path to an fsl flirt matrix or an flirt matrix already
        imported.

    Returns
    -------
        numpy.float64
            squared norm of the translation + squared Frobenium norm of the
            difference between other affine transformations and the identity
    """
    if isinstance(flirt_mat, np.ndarray):
        mat = flirt_mat
    elif isinstance(flirt_mat, str):
        if os.path.exists(flirt_mat):
            mat = np.loadtxt(flirt_mat)
        else:
            raise IOError("ERROR norm_transformation: " + flirt_mat +
                          " file does not exist")
    else:
        raise TypeError("ERROR norm_transformation: flirt_mat should be" +
                        " either a str or a numpy.ndarray matrix")

    if mat.shape != (4, 4):
        raise ValueError("ERROR norm_transformation: the matrix should be 4x4")

    # Translation vector
    translation = mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_affine_transform = mat[0:3, 0:3]
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_affine_transform - np.identity(3), 'fro')
    return pow(tr_norm, 2) + pow(affine_norm, 2)


def template_convergence(mat_file, mat_type='flirt',
                         convergence_threshold=np.finfo(np.float64).eps):
    """

    Parameters
    ----------
    mat_file: str
        path to an fsl flirt matrix
    mat_type: str
        'flirt'(default), 'ants'
        The type of matrix used to represent the transformations
    convergence_threshold: float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.

    Returns
    -------

    """
    if mat_type == 'flirt':
        translation, oth_transform = read_flirt_mat(mat_file)
    elif mat_type == 'ants':
        translation, oth_transform = read_ants_mat(mat_file)
    else:
        raise ValueError("ERROR template_convergence: this matrix type does " +
                         "not exist")
    distance = norm_transformations(translation, oth_transform)

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
    output_folder: str
        path to the output folder (the folder must already exist)
    init_reg: nipype.MapNode
        (default None so no initial registration is performed)
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

    if init_reg is not None:
        init_reg.run()
        image_list = init_reg.inputs.out_file
        mat_list = init_reg.inputs.out_matrix_file
        # test if every transformation matrix has reached the convergence
        convergence_list = [template_convergence(
            mat, 'flirt', convergence_threshold) for mat in mat_list]
        converged = all(convergence_list)
    else:
        image_list = img_list
        converged = False

    tmp_template = os.path.join(output_folder, 'tmp_template.nii.gz')

    while not converged:
        tmp_template = create_temporary_template(image_list,
                                                 out_path=tmp_template,
                                                 avg_method=avg_method)
        reg_list_node = register_img_list(image_list,
                                          ref_img=tmp_template,
                                          output_folder=output_folder,
                                          dof=dof,
                                          interp=interp,
                                          cost=cost)
        reg_list_node.run()

        image_list = reg_list_node.inputs.out_file
        mat_list = reg_list_node.inputs.out_matrix_file
        # test if every transformation matrix has reached the convergence
        convergence_list = [template_convergence(
            mat, 'flirt', convergence_threshold) for mat in mat_list]
        converged = all(convergence_list)

    template = tmp_template
    return template


# multithread + dipy version of the freesurfer longitudinal template creation

# http://nipy.org/dipy/reference/dipy.align.html#dipy.align.imaffine.transform_centers_of_mass
# as the first step to ensure a minimal alignment between the images

def center_align(image, reference):
    from dipy.align.imaffine import transform_centers_of_mass

    img = nifti_image_input(image)
    ref = nifti_image_input(reference)

    moving = img.get_data()
    static = ref.get_data()

    static_grid2world = img.affine
    moving_grid2world = ref.affine

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    transformed = c_of_mass.transform(moving)
    
    return transformed, c_of_mass.affine


def affine_registration(image, reference, init_affine):
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     AffineMap,
                                     MutualInformationMetric,
                                     AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D,
                                       RigidTransform3D,
                                       AffineTransform3D)

    img = nifti_image_input(image)
    ref = nifti_image_input(reference)

    moving = img.get_data()
    static = ref.get_data()

    static_grid2world = img.affine
    moving_grid2world = ref.affine

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = init_affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transformed = rigid.transform(moving)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    transformed = affine.transform(moving)

    out_image = nib.Nifti1Image(transformed, static_grid2world)

    return out_image





def squareNumber(n):
    return n ** 2


# function to be mapped over
def calculateParallel(numbers, threads=2):
    pool = ThreadPool(threads)
    results = pool.map(squareNumber, numbers)
    pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    squaredNumbers = calculateParallel(numbers, 4)
    for n in squaredNumbers:
        print(n)