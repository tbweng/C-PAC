
import os
import ntpath
import numpy as np
import six
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import random

import nibabel as nib
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype import MapNode
from CPAC.utils.nifti_utils import nifti_image_input
from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


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


def read_mat(input_mat):
    if isinstance(input_mat, np.ndarray):
        mat = input_mat
    elif isinstance(input_mat, str):
        if os.path.exists(input_mat):
            mat = np.loadtxt(input_mat)
        else:
            raise IOError("ERROR norm_transformation: " + input_mat +
                          " file does not exist")
    else:
        raise TypeError("ERROR norm_transformation: input_mat should be" +
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
        np.asarray([nifti_image_input(img).get_data() for img in img_list]), 0)

    nii = nib.Nifti1Image(avg_data, nifti_image_input(img_list[0]).affine)
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


def norm_transformation(input_mat):
    """
    Calculate the squared norm of the translation + squared Frobenium norm
    of the difference between other affine transformations and the identity
    from an fsl FLIRT transformation matrix
    Parameters
    ----------
    input_mat: str or numpy.ndarray
        Either the path to text file matrix or a matrix already imported.

    Returns
    -------
        numpy.float64
            squared norm of the translation + squared Frobenius norm of the
            difference between other affine transformations and the identity
    """
    if isinstance(input_mat, np.ndarray):
        mat = input_mat
    elif isinstance(input_mat, str):
        if os.path.exists(input_mat):
            mat = np.loadtxt(input_mat)
        else:
            raise IOError("ERROR norm_transformation: " + input_mat +
                          " file does not exist")
    else:
        raise TypeError("ERROR norm_transformation: input_mat should be" +
                        " either a str (file_path) or a numpy.ndarray matrix")

    if mat.shape != (4, 4):
        raise ValueError("ERROR norm_transformation: the matrix should be 4x4")

    # Translation vector
    translation = mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_affine_transform = mat[0:3, 0:3]
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_affine_transform - np.identity(3), 'fro')
    return pow(tr_norm, 2) + pow(affine_norm, 2)


def template_convergence(mat_file, mat_type='matrix',
                         convergence_threshold=np.finfo(np.float64).eps):
    """
    Calculate the distance between transformation matrix with a matrix of no
    transformation
    Parameters
    ----------
    mat_file: str
        path to an fsl flirt matrix
    mat_type: str
        'matrix'(default), 'ITK'
        The type of matrix used to represent the transformations
    convergence_threshold: float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.

    Returns
    -------

    """
    if mat_type == 'matrix':
        translation, oth_transform = read_mat(mat_file)
    elif mat_type == 'ITK':
        translation, oth_transform = read_ants_mat(mat_file)
    else:
        raise ValueError("ERROR template_convergence: this matrix type does " +
                         "not exist")
    distance = norm_transformations(translation, oth_transform)

    return abs(distance) <= convergence_threshold


def template_creation_flirt(img_list, output_folder,
                            init_reg=MapNode, avg_method='median', dof=12,
                            interp='trilinear', cost='corratio',
                            mat_type='matrix',
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
    mat_type: str
        'matrix'(default), 'ITK'
        The type of matrix used to represent the transformations
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
            mat, mat_type, convergence_threshold) for mat in mat_list]
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
            mat, mat_type, convergence_threshold) for mat in mat_list]
        converged = all(convergence_list)

    template = tmp_template
    return template


def dipy_registration(image, reference, transform_mode='affine',
                      init_affine=None,
                      interp='linear',
                      nbins=32,
                      sampling_prop=None,
                      level_iters=[10000, 1000, 100],
                      sigmas=[3.0, 1.0, 0.0],
                      factors=[4, 2, 1]):
    """
    Calculate and apply the transformations to register a given image to a
    reference image using the dipy.align.imaffine functions.
    Ref: https://github.com/nipy/dipy/blob/master/dipy/align/imaffine.py
    Parameters
    ----------
    image: str or nibabel.nifti1.Nifti1Image
        path to the nifti file or the image already loaded through nibabel to
        be registered to the reference image
    reference: str or nibabel.nifti1.Nifti1Image
        path to the nifti file or the image already loaded through nibabel to
        which the image will be registered
    init_affine : array, shape (dim+1, dim+1), optional
        the pre-aligning matrix (an affine transform) that roughly aligns
        the moving image towards the static image. If None, no
        pre-alignment is performed. If a pre-alignment matrix is available,
        it is recommended to provide this matrix as `starting_affine`
        instead of manually transforming the moving image to reduce
        interpolation artifacts. The default is None, implying no
        pre-alignment is performed.
    interp : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    transform_mode: str or list of int or boolean, optional
        the different transformations to be used to register the images.
        "c_of_mass" will only align the images on the centers of mass
        "translation" will first align the centers of mass and then calculate
        the translation needed to register the images
        "rigid" does the two first calculations and then the rigid
        transformations.
        "affine" does the three precedent calculations and then calculate the
        affine transformations to register the images.
        You can also provide a list of int or boolean to select which
        transformations will be calculated for the registration.
        [0,1,1,0] or [0,1,1] will only calculate the translation and rigid
        transformation and skip the center of mass alignment and affine.
        'c_of_mass': [1],
        'translation': [1, 1],
        'rigid': [1, 1, 1],
        'affine': [1, 1, 1, 1]
    nbins : int, optional
        the number of bins to be used for computing the intensity
        histograms. The default is 32.
    sampling_prop: None or float in interval (0, 1], optional
        There are two types of sampling: dense and sparse. Dense sampling
        uses all voxels for estimating the (joint and marginal) intensity
        histograms, while sparse sampling uses a subset of them. If
        `sampling_proportion` is None, then dense sampling is
        used. If `sampling_proportion` is a floating point value in (0,1]
        then sparse sampling is used, where `sampling_proportion`
        specifies the proportion of voxels to be used. The default is
        None.
    level_iters : sequence, optional
        the number of iterations at each scale of the scale space.
        `level_iters[0]` corresponds to the coarsest scale,
        `level_iters[-1]` the finest, where n is the length of the
        sequence. By default, a 3-level scale space with iterations
        sequence equal to [10000, 1000, 100] will be used.
    sigmas : sequence of floats, optional
        custom smoothing parameter to build the scale space (one parameter
        for each scale). By default, the sequence of sigmas will be
        [3, 1, 0].
    factors : sequence of floats, optional
        custom scale factors to build the scale space (one factor for each
        scale). By default, the sequence of factors will be [4, 2, 1].

    Returns
    -------
    out_image, transformation_matrix: nibabel.Nifti1Image, 4x4 numpy.array
        the image registered to the reference, the transformation matrix of
        the registration
    """

    tr_dict = {'c_of_mass': [1],
               'translation': [1, 1],
               'rigid': [1, 1, 1],
               'affine': [1, 1, 1, 1]}
    if isinstance(transform_mode, list):
        if len(transform_mode) != 0:
            tr_mode = transform_mode
        else:
            raise ValueError("transform_mode is an empty list")
    elif isinstance(transform_mode, six.string_types):
        if transform_mode in tr_dict.keys():
            tr_mode = tr_dict[transform_mode]
        else:
            raise ValueError(transform_mode + " is not in the possible " +
                             "transformations, please choose between: " +
                             str([k for k in tr_dict.keys()]))
    else:
        raise TypeError("transform_mode can be either a string, a list or an " +
                        "int/long")

    if not any(tr_mode):
        print("dipy_registration didn't do anything " +
              " (transform_mode == [0, 0, 0, 0])")
        return nifti_image_input(image), np.eye(4, 4)

    if init_affine is not None and tr_mode[0]:
        print("init_affine will be ignored as transform_centers_of_mass " +
              "doesn't take initial transformation")

    img = nifti_image_input(image)
    ref = nifti_image_input(reference)

    moving = img.get_data()
    static = ref.get_data()

    static_grid2world = img.affine
    moving_grid2world = ref.affine
    # Used only if transform_centers_of_mass is not used
    transformation_matrix = init_affine

    if len(tr_mode) >= len(tr_dict['c_of_mass']) and tr_mode[0]:
        print("Calculating Center of mass")
        c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                              moving, moving_grid2world)

        transformation_matrix = c_of_mass.affine

        transformed = c_of_mass.transform(moving, interp=interp)

    # initialization of the AffineRegistration object used in any of the other
    # transformation
    if len(tr_mode) >= len(tr_dict['translation']):
        metric = MutualInformationMetric(nbins, sampling_prop)

        affreg = AffineRegistration(metric=metric,
                                    level_iters=level_iters,
                                    sigmas=sigmas,
                                    factors=factors)

    if len(tr_mode) >= len(tr_dict['translation']) and tr_mode[1]:
        print("Calculating Translation")
        transform = TranslationTransform3D()
        params0 = None
        translation = affreg.optimize(static, moving, transform, params0,
                                      static_grid2world, moving_grid2world,
                                      starting_affine=transformation_matrix)
        transformation_matrix = translation.affine
        transformed = translation.transform(moving, interp=interp)

    if len(tr_mode) >= len(tr_dict['rigid']) and tr_mode[2]:
        print("Calculating Rigid")
        transform = RigidTransform3D()
        params0 = None

        rigid = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=transformation_matrix)
        transformation_matrix = rigid.affine

        transformed = rigid.transform(moving, interp=interp)

    if len(tr_mode) >= len(tr_dict['affine']) and tr_mode[3]:
        print("Calculating Affine")
        transform = AffineTransform3D()
        params0 = None

        affine = affreg.optimize(static, moving, transform, params0,
                                 static_grid2world, moving_grid2world,
                                 starting_affine=transformation_matrix)

        transformation_matrix = affine.affine

        transformed = affine.transform(moving, interp=interp)

    # Create an image with the transformed data and the affine of the reference
    out_image = nib.Nifti1Image(transformed, static_grid2world)

    return out_image, transformation_matrix


def parallel_dipy_list_reg(img_list, reference,
                           transform_mode='affine',
                           init_affine_list=None,
                           interp='linear',
                           nbins=32,
                           sampling_prop=None,
                           level_iters=[10000, 1000, 100],
                           sigmas=[3.0, 1.0, 0.0],
                           factors=[4, 2, 1], threads=2):
    """
    Register a list of images to a reference image using the
    dipy.align.imaffine functions
    Parameters
    ----------
    img_list: list of str or list of nibabel.Nifti1Image
        list of images to be registered
    reference: str or nibabel.nifti1.Nifti1Image
        path to the nifti file or the image already loaded through nibabel to
        which the image will be registered
    transform_mode: str or list of int or boolean, optional
        the different transformations to be used to register the images.
        "c_of_mass" will only align the images on the centers of mass
        "translation" will first align the centers of mass and then calculate
        the translation needed to register the images
        "rigid" does the two first calculations and then the rigid
        transformations.
        "affine" does the three precedent calculations and then calculate the
        affine transformations to register the images.
        You can also provide a list of int or boolean to select which
        transformations will be calculated for the registration.
        [0,1,1,0] or [0,1,1] will only calculate the translation and rigid
        transformation and skip the center of mass alignment and affine.
    init_affine_list : list of arrays, optional
        the pre-aligning matrix (an affine transform) that roughly aligns
        the moving image towards the static image. If None, no
        pre-alignment is performed. If a pre-alignment matrix is available,
        it is recommended to provide this matrix as `starting_affine`
        instead of manually transforming the moving image to reduce
        interpolation artifacts. The default is None, implying no
        pre-alignment is performed.
    interp : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    nbins : int, optional
        the number of bins to be used for computing the intensity
        histograms. The default is 32.
    sampling_prop: None or float in interval (0, 1], optional
        There are two types of sampling: dense and sparse. Dense sampling
        uses all voxels for estimating the (joint and marginal) intensity
        histograms, while sparse sampling uses a subset of them. If
        `sampling_proportion` is None, then dense sampling is
        used. If `sampling_proportion` is a floating point value in (0,1]
        then sparse sampling is used, where `sampling_proportion`
        specifies the proportion of voxels to be used. The default is
        None.
    level_iters : sequence, optional
        the number of iterations at each scale of the scale space.
        `level_iters[0]` corresponds to the coarsest scale,
        `level_iters[-1]` the finest, where n is the length of the
        sequence. By default, a 3-level scale space with iterations
        sequence equal to [10000, 1000, 100] will be used.
    sigmas : sequence of floats, optional
        custom smoothing parameter to build the scale space (one parameter
        for each scale). By default, the sequence of sigmas will be
        [3, 1, 0].
    factors : sequence of floats, optional
        custom scale factors to build the scale space (one factor for each
        scale). By default, the sequence of factors will be [4, 2, 1].
    threads: int
        (default 2) number of threads

    Returns
    -------
    results: list of (out_image, transformation_matrix)
        nibabel.Nifti1Image, 4x4 numpy.array
        the images registered to the reference, the transformation matrices of
        the registrations
    """

    pool = ThreadPool(threads)

    if init_affine_list is not None:
        if len(init_affine_list) != len(img_list):
            raise ValueError("The list of images and affine matrices don't " +
                             "have the same size")
        temp = partial(dipy_registration,
                       interp=interp,
                       nbins=nbins,
                       sampling_prop=sampling_prop,
                       level_iters=level_iters,
                       sigmas=sigmas,
                       factors=factors)
        img_num = len(img_list)
        list_args = zip(img_list,
                        [reference] * img_num,
                        [transform_mode] * img_num,
                        init_affine_list)
        results = pool.starmap(temp, list_args)
    else:
        temp = partial(dipy_registration,
                       reference=reference,
                       transform_mode=transform_mode,
                       init_affine=init_affine_list,
                       interp=interp,
                       nbins=nbins,
                       sampling_prop=sampling_prop,
                       level_iters=level_iters,
                       sigmas=sigmas,
                       factors=factors)
        results = pool.map(temp, img_list)
    pool.close()
    pool.join()
    # results should be a list of tuples of registered images and their
    # transformation matrix
    return results


def template_creation_dipy(img_list, output_folder,
                           avg_method='median',
                           transform_mode='affine',
                           interp='linear',
                           init_method='random_image',
                           nbins=32,
                           sampling_prop=None,
                           level_iters=[10000, 1000, 100],
                           sigmas=[3.0, 1.0, 0.0],
                           factors=[4, 2, 1],
                           convergence_threshold=np.finfo(np.float64).eps,
                           threads=2):
    """
    Use the Reuter et al. 2012 (NeuroImage) doi:10.1016/j.neuroimage.2012.02.084
    principle to calculate a template of longitudinal data. The average of the
    image list is calculated and then each image is registered to this image.
    This process is repeated on the registered images until the distance between
    the transformations and no transformation is smaller than the convergence
    threshold.
    Parameters
    ----------
    img_list: list of str or list of nibabel.Nifti1Image
        list of images to be registered
    output_folder: str
        path to the output folder (the folder must already exist)
    avg_method: str, optional
        function names from numpy library such as 'median', 'mean', 'std' ...
    transform_mode: str or list of int or boolean, optional
        the different transformations to be used to register the images.
        "c_of_mass" will only align the images on the centers of mass
        "translation" will first align the centers of mass and then calculate
        the translation needed to register the images
        "rigid" does the two first calculations and then the rigid
        transformations.
        "affine" does the three precedent calculations and then calculate the
        affine transformations to register the images.
        You can also provide a list of int or boolean to select which
        transformations will be calculated for the registration.
        [0,1,1,0] or [0,1,1] will only calculate the translation and rigid
        transformation and skip the center of mass alignment and affine.
    interp : string, either 'linear' or 'nearest', optional
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    init_method: str, 'random_image' (default), 'None' or None optional
        Method used to initialize the template creation.
        'random_image' will register the images to a randomly selected image
        from the original dataset to speed up the process as suggested in
        Reuter et al. 2012 (NeuroImage) section "Improved template estimation"
        doi:10.1016/j.neuroimage.2012.02.084
    nbins : int, optional
        the number of bins to be used for computing the intensity
        histograms. The default is 32.
    sampling_prop: None or float in interval (0, 1], optional
        There are two types of sampling: dense and sparse. Dense sampling
        uses all voxels for estimating the (joint and marginal) intensity
        histograms, while sparse sampling uses a subset of them. If
        `sampling_proportion` is None, then dense sampling is
        used. If `sampling_proportion` is a floating point value in (0,1]
        then sparse sampling is used, where `sampling_proportion`
        specifies the proportion of voxels to be used. The default is
        None.
    level_iters : sequence, optional
        the number of iterations at each scale of the scale space.
        `level_iters[0]` corresponds to the coarsest scale,
        `level_iters[-1]` the finest, where n is the length of the
        sequence. By default, a 3-level scale space with iterations
        sequence equal to [10000, 1000, 100] will be used.
    sigmas : sequence of floats, optional
        custom smoothing parameter to build the scale space (one parameter
        for each scale). By default, the sequence of sigmas will be
        [3, 1, 0].
    factors : sequence of floats, optional
        custom scale factors to build the scale space (one factor for each
        scale). By default, the sequence of factors will be [4, 2, 1].
    convergence_threshold: float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.
    threads: int
        (default 2) number of threads

    Returns
    -------
    template: str
        path to the final template

    Notes
    -----
    The function can be initialized with a list of images already transformed.
    """
    if not img_list:
        print('ERROR create_temporary_template: image list is empty')

    converged = False

    tmp_template = os.path.join(output_folder, 'tmp_template.nii.gz')

    # First align the center of mass of the images together before the first
    # average. As this is only a translation, the reference image for the
    # center of mass is the first image of the list.

    center_align = parallel_dipy_list_reg(img_list,
                                          interp=interp,
                                          reference=img_list[0],
                                          transform_mode='c_of_mass')

    image_list = [res[0] for res in center_align]

    if init_method == 'random_image':

        index = random.randrange(len(image_list))

        res_list_reg = parallel_dipy_list_reg(img_list,
                                              reference=image_list[index],
                                              transform_mode=transform_mode,
                                              init_affine_list=None,
                                              nbins=nbins,
                                              sampling_prop=sampling_prop,
                                              level_iters=level_iters,
                                              sigmas=sigmas,
                                              factors=factors, threads=threads)
        image_list = [res[0] for res in res_list_reg]

    while not converged:
        tmp_template = create_temporary_template(image_list,
                                                 out_path=tmp_template,
                                                 avg_method=avg_method)
        res_list_reg = parallel_dipy_list_reg(img_list,
                                              reference=tmp_template,
                                              transform_mode=transform_mode,
                                              init_affine_list=None,
                                              nbins=nbins,
                                              sampling_prop=sampling_prop,
                                              level_iters=level_iters,
                                              sigmas=sigmas,
                                              factors=factors, threads=threads)

        image_list = [res[0] for res in res_list_reg]
        mat_list = [res[1] for res in res_list_reg]
        # test if every transformation matrix has reached the convergence
        convergence_list = [template_convergence(
            mat, 'matrix', convergence_threshold) for mat in mat_list]
        print("Average matrices distance " + str(np.mean(convergence_list)))
        converged = all(convergence_list)

    template = tmp_template
    return template
