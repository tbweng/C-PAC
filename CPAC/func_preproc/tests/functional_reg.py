# -*- coding: utf-8 -*-
import os
import ntpath
import numpy as np
import six
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import random
from collections import Iterable
import re
import subprocess
import glob

from CPAC.utils.nifti_utils import nifti_image_input
import nibabel as nib
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype import MapNode
# from dipy.align.imaffine import (transform_centers_of_mass,
#                                  MutualInformationMetric,
#                                  AffineRegistration)
# from dipy.align.transforms import (TranslationTransform3D,
#                                    RigidTransform3D,
#                                    AffineTransform3D)


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
    elif isinstance(input_mat, six.string_types):
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


def register_img_list_mapnode(img_list, ref_img, output_folder, dof=12,
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


def register_img_list(img_list, ref_img, output_folder, dof=12,
                      interp='trilinear', cost='corratio', thread_pool=2):
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
    multiple_linear_reg: list of Node
        outputs.out_file will contain the registered images
        outputs.out_matrix_file will contain the transformation matrices
    """
    if not img_list:
        raise ValueError('ERROR register_img_list: image list is empty')

    # output_folder = os.getcwd()

    output_img_list = [os.path.join(output_folder, ntpath.basename(img))
                       for img in img_list]

    output_mat_list = [os.path.join(output_folder,
                                    str(ntpath.basename(img).split('.')[0])
                                    + '.mat')
                       for img in img_list]

    def flirt_node(img, out_img, out_mat):
        linear_reg = fsl.FLIRT()
        linear_reg.inputs.in_file = img
        linear_reg.inputs.out_file = out_img
        linear_reg.inputs.out_matrix_file = out_mat

        linear_reg.inputs.cost = cost
        linear_reg.inputs.dof = dof
        linear_reg.inputs.interp = interp
        linear_reg.inputs.reference = ref_img

        return linear_reg

    if isinstance(thread_pool, int):
        pool = ThreadPool(thread_pool)
    else:
        pool = thread_pool

    node_list = [flirt_node(img, out_img, out_mat)
                 for (img, out_img, out_mat) in zip(
                 img_list, output_img_list, output_mat_list)]
    pool.map(lambda node: node.run(), node_list)

    if isinstance(thread_pool, int):
        pool.close()
        pool.join()

    return node_list


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
    elif isinstance(input_mat, six.string_types):
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
                            init_reg=None, avg_method='median', dof=12,
                            interp='trilinear', cost='corratio',
                            mat_type='matrix',
                            convergence_threshold=np.finfo(np.float64).eps,
                            thread_pool=2):
    """

    Parameters
    ----------
    img_list: list of str
        list of images paths
    output_folder: str
        path to the output folder (the folder must already exist)
    init_reg: list of Node
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
    if isinstance(thread_pool, int):
        pool = ThreadPool(thread_pool)
    else:
        pool = thread_pool

    if not img_list:
        print('ERROR create_temporary_template: image list is empty')

    if init_reg is not None:
        image_list = [node.inputs.out_file for node in init_reg]
        mat_list = [node.inputs.out_matrix_file for node in init_reg]
        # test if every transformation matrix has reached the convergence
        convergence_list = [template_convergence(
            mat, mat_type, convergence_threshold) for mat in mat_list]
        converged = all(convergence_list)
    else:
        image_list = img_list
        converged = False

    tmp_template = os.path.join(output_folder, 'tmp_template.nii.gz')
    # file1 = open("/Users/cf27246/convergence_test.txt", "w")  # write mode
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

        image_list = [node.inputs.out_file for node in reg_list_node]
        mat_list = [node.inputs.out_matrix_file for node in reg_list_node]
        print(str(mat_list))
        # test if every transformation matrix has reached the convergence
        convergence_list = [template_convergence(
            mat, mat_type, convergence_threshold) for mat in mat_list]
        converged = all(convergence_list)

    #     # DEBUG
    #     norms = [read_mat(mat) for mat in mat_list]
    #     mean_distance = np.mean(np.array([norm_transformations(tr, oth)
    #                                       for (tr, oth) in norms]))
    #     print("Average matrices distance " + str(mean_distance))
    #     file1.write("Average matrices distance" +
    #                 str(mean_distance) + "\n")
    # file1.close()

    if isinstance(thread_pool, int):
        pool.close()
        pool.join()

    template = tmp_template
    return template


# def dipy_registration(image, reference, transform_mode='affine',
#                       init_affine=None,
#                       interp='linear',
#                       nbins=32,
#                       sampling_prop=None,
#                       level_iters=[10000, 1000, 100],
#                       sigmas=[3.0, 1.0, 0.0],
#                       factors=[4, 2, 1]):
#     """
#     Calculate and apply the transformations to register a given image to a
#     reference image using the dipy.align.imaffine functions.
#     Ref: https://github.com/nipy/dipy/blob/master/dipy/align/imaffine.py
#     Parameters
#     ----------
#     image: str or nibabel.nifti1.Nifti1Image
#         path to the nifti file or the image already loaded through nibabel to
#         be registered to the reference image
#     reference: str or nibabel.nifti1.Nifti1Image
#         path to the nifti file or the image already loaded through nibabel to
#         which the image will be registered
#     init_affine : array, shape (dim+1, dim+1), optional
#         the pre-aligning matrix (an affine transform) that roughly aligns
#         the moving image towards the static image. If None, no
#         pre-alignment is performed. If a pre-alignment matrix is available,
#         it is recommended to provide this matrix as `starting_affine`
#         instead of manually transforming the moving image to reduce
#         interpolation artifacts. The default is None, implying no
#         pre-alignment is performed.
#     interp : string, either 'linear' or 'nearest'
#         the type of interpolation to be used, either 'linear'
#         (for k-linear interpolation) or 'nearest' for nearest neighbor
#     transform_mode: str or list of int or boolean, optional
#         the different transformations to be used to register the images.
#         "c_of_mass" will only align the images on the centers of mass
#         "translation" will first align the centers of mass and then calculate
#         the translation needed to register the images
#         "rigid" does the two first calculations and then the rigid
#         transformations.
#         "affine" does the three precedent calculations and then calculate the
#         affine transformations to register the images.
#         You can also provide a list of int or boolean to select which
#         transformations will be calculated for the registration.
#         [0,1,1,0] or [0,1,1] will only calculate the translation and rigid
#         transformation and skip the center of mass alignment and affine.
#         'c_of_mass': [1],
#         'translation': [1, 1],
#         'rigid': [1, 1, 1],
#         'affine': [1, 1, 1, 1]
#     nbins : int, optional
#         the number of bins to be used for computing the intensity
#         histograms. The default is 32.
#     sampling_prop: None or float in interval (0, 1], optional
#         There are two types of sampling: dense and sparse. Dense sampling
#         uses all voxels for estimating the (joint and marginal) intensity
#         histograms, while sparse sampling uses a subset of them. If
#         `sampling_proportion` is None, then dense sampling is
#         used. If `sampling_proportion` is a floating point value in (0,1]
#         then sparse sampling is used, where `sampling_proportion`
#         specifies the proportion of voxels to be used. The default is
#         None.
#     level_iters : sequence, optional
#         the number of iterations at each scale of the scale space.
#         `level_iters[0]` corresponds to the coarsest scale,
#         `level_iters[-1]` the finest, where n is the length of the
#         sequence. By default, a 3-level scale space with iterations
#         sequence equal to [10000, 1000, 100] will be used.
#     sigmas : sequence of floats, optional
#         custom smoothing parameter to build the scale space (one parameter
#         for each scale). By default, the sequence of sigmas will be
#         [3, 1, 0].
#     factors : sequence of floats, optional
#         custom scale factors to build the scale space (one factor for each
#         scale). By default, the sequence of factors will be [4, 2, 1].
#
#     Returns
#     -------
#     out_image, transformation_matrix: nibabel.Nifti1Image, 4x4 numpy.array
#         the image registered to the reference, the transformation matrix of
#         the registration
#     """
#
#     tr_dict = {'c_of_mass': [1],
#                'translation': [1, 1],
#                'rigid': [1, 1, 1],
#                'affine': [1, 1, 1, 1]}
#     if isinstance(transform_mode, list):
#         if len(transform_mode) != 0:
#             tr_mode = transform_mode
#         else:
#             raise ValueError("transform_mode is an empty list")
#     elif isinstance(transform_mode, six.string_types):
#         if transform_mode in tr_dict.keys():
#             tr_mode = tr_dict[transform_mode]
#         else:
#             raise ValueError(transform_mode + " is not in the possible " +
#                              "transformations, please choose between: " +
#                              str([k for k in tr_dict.keys()]))
#     else:
#         raise TypeError("transform_mode can be either a string, a list or an " +
#                         "int/long")
#
#     if not any(tr_mode):
#         print("dipy_registration didn't do anything " +
#               " (transform_mode == [0, 0, 0, 0])")
#         return nifti_image_input(image), np.eye(4, 4)
#
#     if init_affine is not None and tr_mode[0]:
#         print("init_affine will be ignored as transform_centers_of_mass " +
#               "doesn't take initial transformation")
#
#     img = nifti_image_input(image)
#     ref = nifti_image_input(reference)
#
#     moving = img.get_data()
#     static = ref.get_data()
#
#     static_grid2world = img.affine
#     moving_grid2world = ref.affine
#     # Used only if transform_centers_of_mass is not used
#     transformation_matrix = init_affine
#
#     if len(tr_mode) >= len(tr_dict['c_of_mass']) and tr_mode[0]:
#         print("Calculating Center of mass")
#         c_of_mass = transform_centers_of_mass(static, static_grid2world,
#                                               moving, moving_grid2world)
#
#         transformation_matrix = c_of_mass.affine
#
#         transformed = c_of_mass.transform(moving, interp=interp)
#
#     # initialization of the AffineRegistration object used in any of the other
#     # transformation
#     if len(tr_mode) >= len(tr_dict['translation']):
#         metric = MutualInformationMetric(nbins, sampling_prop)
#
#         affreg = AffineRegistration(metric=metric,
#                                     level_iters=level_iters,
#                                     sigmas=sigmas,
#                                     factors=factors)
#
#     if len(tr_mode) >= len(tr_dict['translation']) and tr_mode[1]:
#         print("Calculating Translation")
#         transform = TranslationTransform3D()
#         params0 = None
#         translation = affreg.optimize(static, moving, transform, params0,
#                                       static_grid2world, moving_grid2world,
#                                       starting_affine=transformation_matrix)
#         transformation_matrix = translation.affine
#         transformed = translation.transform(moving, interp=interp)
#
#     if len(tr_mode) >= len(tr_dict['rigid']) and tr_mode[2]:
#         print("Calculating Rigid")
#         transform = RigidTransform3D()
#         params0 = None
#
#         rigid = affreg.optimize(static, moving, transform, params0,
#                                 static_grid2world, moving_grid2world,
#                                 starting_affine=transformation_matrix)
#         transformation_matrix = rigid.affine
#
#         transformed = rigid.transform(moving, interp=interp)
#
#     if len(tr_mode) >= len(tr_dict['affine']) and tr_mode[3]:
#         print("Calculating Affine")
#         transform = AffineTransform3D()
#         params0 = None
#
#         affine = affreg.optimize(static, moving, transform, params0,
#                                  static_grid2world, moving_grid2world,
#                                  starting_affine=transformation_matrix)
#
#         transformation_matrix = affine.affine
#
#         transformed = affine.transform(moving, interp=interp)
#
#     # Create an image with the transformed data and the affine of the reference
#     out_image = nib.Nifti1Image(transformed, static_grid2world)
#
#     return out_image, transformation_matrix
#
#
# def parallel_dipy_list_reg(img_list, reference,
#                            transform_mode='affine',
#                            init_affine_list=None,
#                            interp='linear',
#                            nbins=32,
#                            sampling_prop=None,
#                            level_iters=[10000, 1000, 100],
#                            sigmas=[3.0, 1.0, 0.0],
#                            factors=[4, 2, 1],
#                            thread_pool=2):
#     """
#     Register a list of images to a reference image using the
#     dipy.align.imaffine functions
#     Parameters
#     ----------
#     img_list: list of str or list of nibabel.Nifti1Image
#         list of images to be registered
#     reference: str or nibabel.nifti1.Nifti1Image
#         path to the nifti file or the image already loaded through nibabel to
#         which the image will be registered
#     transform_mode: str or list of int or boolean, optional
#         the different transformations to be used to register the images.
#         "c_of_mass" will only align the images on the centers of mass
#         "translation" will first align the centers of mass and then calculate
#         the translation needed to register the images
#         "rigid" does the two first calculations and then the rigid
#         transformations.
#         "affine" does the three precedent calculations and then calculate the
#         affine transformations to register the images.
#         You can also provide a list of int or boolean to select which
#         transformations will be calculated for the registration.
#         [0,1,1,0] or [0,1,1] will only calculate the translation and rigid
#         transformation and skip the center of mass alignment and affine.
#     init_affine_list : list of arrays, optional
#         the pre-aligning matrix (an affine transform) that roughly aligns
#         the moving image towards the static image. If None, no
#         pre-alignment is performed. If a pre-alignment matrix is available,
#         it is recommended to provide this matrix as `starting_affine`
#         instead of manually transforming the moving image to reduce
#         interpolation artifacts. The default is None, implying no
#         pre-alignment is performed.
#     interp : string, either 'linear' or 'nearest'
#         the type of interpolation to be used, either 'linear'
#         (for k-linear interpolation) or 'nearest' for nearest neighbor
#     nbins : int, optional
#         the number of bins to be used for computing the intensity
#         histograms. The default is 32.
#     sampling_prop: None or float in interval (0, 1], optional
#         There are two types of sampling: dense and sparse. Dense sampling
#         uses all voxels for estimating the (joint and marginal) intensity
#         histograms, while sparse sampling uses a subset of them. If
#         `sampling_proportion` is None, then dense sampling is
#         used. If `sampling_proportion` is a floating point value in (0,1]
#         then sparse sampling is used, where `sampling_proportion`
#         specifies the proportion of voxels to be used. The default is
#         None.
#     level_iters : sequence, optional
#         the number of iterations at each scale of the scale space.
#         `level_iters[0]` corresponds to the coarsest scale,
#         `level_iters[-1]` the finest, where n is the length of the
#         sequence. By default, a 3-level scale space with iterations
#         sequence equal to [10000, 1000, 100] will be used.
#     sigmas : sequence of floats, optional
#         custom smoothing parameter to build the scale space (one parameter
#         for each scale). By default, the sequence of sigmas will be
#         [3, 1, 0].
#     factors : sequence of floats, optional
#         custom scale factors to build the scale space (one factor for each
#         scale). By default, the sequence of factors will be [4, 2, 1].
#     threads: int
#         (default 2) number of threads
#
#     Returns
#     -------
#     results: list of (out_image, transformation_matrix)
#         nibabel.Nifti1Image, 4x4 numpy.array
#         the images registered to the reference, the transformation matrices of
#         the registrations
#     """
#
#     if isinstance(thread_pool, int):
#         pool = ThreadPool(thread_pool)
#     else:
#         pool = thread_pool
#
#     if init_affine_list is not None:
#         if len(init_affine_list) != len(img_list):
#             raise ValueError("The list of images and affine matrices don't " +
#                              "have the same size")
#         temp = partial(dipy_registration,
#                        interp=interp,
#                        nbins=nbins,
#                        sampling_prop=sampling_prop,
#                        level_iters=level_iters,
#                        sigmas=sigmas,
#                        factors=factors)
#         img_num = len(img_list)
#         list_args = zip(img_list,
#                         [reference] * img_num,
#                         [transform_mode] * img_num,
#                         init_affine_list)
#         results = pool.starmap(temp, list_args)
#     else:
#         temp = partial(dipy_registration,
#                        reference=reference,
#                        transform_mode=transform_mode,
#                        init_affine=init_affine_list,
#                        interp=interp,
#                        nbins=nbins,
#                        sampling_prop=sampling_prop,
#                        level_iters=level_iters,
#                        sigmas=sigmas,
#                        factors=factors)
#         results = pool.map(temp, img_list)
#
#     if isinstance(thread_pool, int):
#         pool.close()
#         pool.join()
#     # results should be a list of tuples of registered images and their
#     # transformation matrix
#     return results
#
#
# def template_creation_dipy(img_list, output_folder,
#                            avg_method='median',
#                            transform_mode='affine',
#                            interp='linear',
#                            init_method='random_image',
#                            nbins=32,
#                            sampling_prop=None,
#                            level_iters=[10000, 1000, 100],
#                            sigmas=[3.0, 1.0, 0.0],
#                            factors=[4, 2, 1],
#                            convergence_threshold=np.finfo(np.float64).eps,
#                            threads=2):
#     """
#     Use the Reuter et al. 2012 (NeuroImage) doi:10.1016/j.neuroimage.2012.02.084
#     principle to calculate a template of longitudinal data. The average of the
#     image list is calculated and then each image is registered to this image.
#     This process is repeated on the registered images until the distance between
#     the transformations and no transformation is smaller than the convergence
#     threshold.
#     Parameters
#     ----------
#     img_list: list of str or list of nibabel.Nifti1Image
#         list of images to be registered
#     output_folder: str
#         path to the output folder (the folder must already exist)
#     avg_method: str, optional
#         function names from numpy library such as 'median', 'mean', 'std' ...
#     transform_mode: str or list of int or boolean, optional
#         the different transformations to be used to register the images.
#         "c_of_mass" will only align the images on the centers of mass
#         "translation" will first align the centers of mass and then calculate
#         the translation needed to register the images
#         "rigid" does the two first calculations and then the rigid
#         transformations.
#         "affine" does the three precedent calculations and then calculate the
#         affine transformations to register the images.
#         You can also provide a list of int or boolean to select which
#         transformations will be calculated for the registration.
#         [0,1,1,0] or [0,1,1] will only calculate the translation and rigid
#         transformation and skip the center of mass alignment and affine.
#     interp : string, either 'linear' or 'nearest', optional
#         the type of interpolation to be used, either 'linear'
#         (for k-linear interpolation) or 'nearest' for nearest neighbor
#     init_method: str, 'random_image' (default), 'None' or None optional
#         Method used to initialize the template creation.
#         'random_image' will register the images to a randomly selected image
#         from the original dataset to speed up the process as suggested in
#         Reuter et al. 2012 (NeuroImage) section "Improved template estimation"
#         doi:10.1016/j.neuroimage.2012.02.084
#     nbins : int, optional
#         the number of bins to be used for computing the intensity
#         histograms. The default is 32.
#     sampling_prop: None or float in interval (0, 1], optional
#         There are two types of sampling: dense and sparse. Dense sampling
#         uses all voxels for estimating the (joint and marginal) intensity
#         histograms, while sparse sampling uses a subset of them. If
#         `sampling_proportion` is None, then dense sampling is
#         used. If `sampling_proportion` is a floating point value in (0,1]
#         then sparse sampling is used, where `sampling_proportion`
#         specifies the proportion of voxels to be used. The default is
#         None.
#     level_iters : sequence, optional
#         the number of iterations at each scale of the scale space.
#         `level_iters[0]` corresponds to the coarsest scale,
#         `level_iters[-1]` the finest, where n is the length of the
#         sequence. By default, a 3-level scale space with iterations
#         sequence equal to [10000, 1000, 100] will be used.
#     sigmas : sequence of floats, optional
#         custom smoothing parameter to build the scale space (one parameter
#         for each scale). By default, the sequence of sigmas will be
#         [3, 1, 0].
#     factors : sequence of floats, optional
#         custom scale factors to build the scale space (one factor for each
#         scale). By default, the sequence of factors will be [4, 2, 1].
#     convergence_threshold: float
#         (numpy.finfo(np.float64).eps (default)) threshold for the convergence
#         The threshold is how different from no transformation is the
#         transformation matrix.
#     threads: int
#         (default 2) number of threads
#
#     Returns
#     -------
#     template: str
#         path to the final template
#
#     Notes
#     -----
#     The function can be initialized with a list of images already transformed.
#     """
#     if not isinstance(img_list, (list,np.ndarray)):
#         raise TypeError("img_list is not a list or a numpy array")
#     if not img_list:
#         raise ValueError('img_list is empty')
#
#     converged = False
#
#     tmp_template = os.path.join(output_folder, 'tmp_template.nii.gz')
#
#     # First align the center of mass of the images together before the first
#     # average. As this is only a translation, the reference image for the
#     # center of mass is the first image of the list.
#
#     index = random.randrange(len(img_list))
#
#     pool = ThreadPool(threads)
#
#     center_align = parallel_dipy_list_reg(img_list,
#                                           interp=interp,
#                                           reference=img_list[index],
#                                           transform_mode='c_of_mass',
#                                           thread_pool=pool)
#
#     image_list = [res[0] for res in center_align]
#
#     if init_method == 'random_image':
#
#         res_list_reg = parallel_dipy_list_reg(img_list,
#                                               reference=image_list[index],
#                                               transform_mode=transform_mode,
#                                               init_affine_list=None,
#                                               nbins=nbins,
#                                               sampling_prop=sampling_prop,
#                                               level_iters=level_iters,
#                                               sigmas=sigmas,
#                                               factors=factors,
#                                               thread_pool=pool)
#         image_list = [res[0] for res in res_list_reg]
#
#     file1 = open("/Users/cf27246/convergence_test.txt", "w")  # write mode
#     while not converged:
#         tmp_template = create_temporary_template(image_list,
#                                                  out_path=tmp_template,
#                                                  avg_method=avg_method)
#         res_list_reg = parallel_dipy_list_reg(img_list,
#                                               reference=tmp_template,
#                                               transform_mode=transform_mode,
#                                               init_affine_list=None,
#                                               nbins=nbins,
#                                               sampling_prop=sampling_prop,
#                                               level_iters=level_iters,
#                                               sigmas=sigmas,
#                                               factors=factors,
#                                               thread_pool=pool)
#
#         image_list = [res[0] for res in res_list_reg]
#         mat_list = [res[1] for res in res_list_reg]
#         # test if every transformation matrix has reached the convergence
#         convergence_list = [template_convergence(
#             mat, 'matrix', convergence_threshold) for mat in mat_list]
#         # DEBUG
#         norms = [read_mat(mat) for mat in mat_list]
#         mean_distance = np.mean(np.array([norm_transformations(tr, oth)
#                                           for (tr, oth) in norms]))
#         print("Average matrices distance " + str(mean_distance))
#         file1.write("Average matrices distance" +
#                     str(mean_distance) + "\n")
#
#         converged = all(convergence_list)
#
#     file1.close()
#
#     template = tmp_template
#     return template
#
#
# def longitudinal_template_creation_node(
#         wf_name='longitudinal_template_creation_node', num_threads=1):
#     create_template = pe.Workflow(name=wf_name)
#
#     inputspec = pe.Node(util.IdentityInterface(
#         fields=['img_list',
#                 'output_folder',
#                 'avg_method',
#                 'transform_mode',
#                 'interp',
#                 'init_method',
#                 'nbins',
#                 'sampling_prop',
#                 'level_iters',
#                 'sigmas',
#                 'factors',
#                 'convergence_threshold']))
#
#     outputspec = pe.Node(util.IdentityInterface(
#         fields=['template']), name='outputspec')
#
#     # average the images
#
#     calc_longitudinal_template = \
#         pe.Node(interface=util.Function(input_names=[['img_list',
#                                                       'output_folder',
#                                                       'avg_method',
#                                                       'transform_mode',
#                                                       'interp',
#                                                       'init_method',
#                                                       'nbins',
#                                                       'sampling_prop',
#                                                       'level_iters',
#                                                       'sigmas',
#                                                       'factors',
#                                                       'convergence_threshold',
#                                                       'threads']],
#                                         output_names=['warp_list',
#                                                       'warped_image'],
#                                         function=template_creation_dipy),
#                 name='calc_ants_warp')
#     calc_longitudinal_template.inputs.threads = num_threads
#     return calc_longitudinal_template


def format_ants_param(option, transform_list, expected_type):
    l_opt = len(option)
    l_tr = len(transform_list)
    # meaning we expect to have a list of lists
    if expected_type == list:
        if isinstance(option[0], list) and l_opt == l_tr:
            return option
        # option is something like [[elem1, elem2 ...]]
        elif isinstance(option[0], list) and l_opt == 1:
            return option * l_tr
        # option is a simple list [elem1, elem2 ...] but we want a list of lists
        else:
            return [option] * l_tr
    # we expect to have a list of "expected_type" elements
    if isinstance(option, expected_type):
        return [option] * l_tr
    elif isinstance(option, list):
        if l_opt == 1:
            return option * l_tr
    # option is in the right format
    if l_opt == l_tr:
        return option
    # option is in the simplified format
    elif l_opt == 1:
        return option * l_tr
    else:
        raise ValueError("The list-shaped parameter should either have")


# def ants_affine_reg(fixed_image,
#                     moving_image,
#                     output_transform_prefix,
#                     transforms,
#                     transform_parameters,
#                     number_of_iterations,
#                     dimension=3,
#                     metric,
#                     metric_weight,
#                     radius_or_number_of_bins,
#                     sampling_strategy,
#                     sampling_percentage,
#                     convergence_threshold=np.finfo(np.float64).eps,
#                     convergence_window_size,
#                     smoothing_sigmas,
#                     shrink_factors,
#                     threads=2):
#     reg.inputs.write_composite_transform = True
#     reg.inputs.collapse_output_transforms = False
#     reg.inputs.initialize_transforms_per_stage = False
#     # We always use voxels for every transformation
#     reg.inputs.sigma_units = ['vox'] * len(smoothing_sigmas)
#     reg.inputs.use_estimate_learning_rate_once = [True, True]
#     reg.inputs.use_histogram_matching = [True, True]
#
# reg.inputs.fixed_image = 'fixed1.nii'
# reg.inputs.moving_image = 'moving1.nii'
# reg.inputs.output_transform_prefix = "output_"
# reg.inputs.transforms = ['Affine', 'SyN']
# reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
# reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
# reg.inputs.dimension = 3
# reg.inputs.write_composite_transform = True
# reg.inputs.collapse_output_transforms = False
# reg.inputs.initialize_transforms_per_stage = False
# reg.inputs.metric = ['CC'] * 2
# reg.inputs.metric_weight = [1] * 2  # Default (value ignored currently by ANTs)
# reg.inputs.radius_or_number_of_bins = [32] * 2
# reg.inputs.sampling_strategy = ['Random']
# reg.inputs.sampling_percentage = [0.05, None]
# reg.inputs.convergence_threshold = [1.e-8, 1.e-9]
# reg.inputs.convergence_window_size = [20] * 2
# reg.inputs.smoothing_sigmas = [[1, 0], [2, 1, 0]]
# reg.inputs.sigma_units = ['vox'] * 2
# reg.inputs.shrink_factors = [[2, 1], [3, 2, 1]]
# reg.inputs.use_estimate_learning_rate_once = [True, True]
# reg.inputs.use_histogram_matching = [True, True]


# class Metric(object):
#     common_metrics = {'cc': 'CC',
#                       'mi': 'MI',
#                       'mattes': 'Mattes',
#                       'meansquares': 'MeanSquares',
#                       'demons': 'Demons',
#                       'gc': 'GC'}
#     samp_strategies = {'none': 'None',
#                        'regular': 'Regular',
#                        'random': 'Random'}
#
#     def __init__(self, metric,
#                  fixed_img,
#                  moving_img,
#                  metric_weight,
#                  radius_or_nb_bins=None,
#                  sampling_strategy='None',
#                  sampling_percentage=1.0):
#         if metric.lower() not in self.common_metrics.keys():
#             raise ValueError("Unknown registration metric type: " + metric)
#         if metric.lower() in ['cc', 'mi', 'mattes'] \
#                 and radius_or_nb_bins is None:
#             raise ValueError("radius_or_nb_bins cannot be None for CC, MI "+
#                              "or Mattes")
#         if not os.path.exists(fixed_img):
#             raise ValueError(fixed_img + " is not an existing file")
#         if not os.path.exists(moving_img):
#             raise ValueError(moving_img + " is not an existing file")
#         if sampling_strategy is None:
#             sampling_strat = 'None'
#         elif sampling_strategy.lower() not in self.samp_strategies.keys():
#             raise ValueError(str(sampling_strategy) +
#                              "can either be 'None', " +
#                              "'Regular' or 'Random'")
#         else:
#             sampling_strat = self.samp_strategies[sampling_strategy.lower()]
#         if not (0.0 <= sampling_percentage <= 1.0):
#             raise ValueError(str(sampling_percentage) + "is a percentage " +
#                              "between 0.0 and 1.0")
#
#         self.out_str = self.common_metrics[metric.lower()] + '['
#         self.out_str = self.out_str + ','.join(
#             [fixed_img,
#              moving_img,
#              str(metric_weight),
#              str(radius_or_nb_bins),
#              sampling_strat,
#              str(float(sampling_percentage))])
#
#         self.out_str = self.out_str + ']'


class Metric(object):
    metrics = {'cc': 'CC',
               'mi': 'MI',
               'mattes': 'Mattes',
               'meansquares': 'MeanSquares',
               'demons': 'Demons',
               'gc': 'GC',
               'icp': 'ICP',
               'pse': 'PSE',
               'jhct': 'JHCT',
               'igdm': 'IGDM'}
    # Minimum number of arguments expected for the transformations
    expected_param = {
        'cc': 4,
        'mi': 4,
        'mattes': 4,
        'meansquares': 3,
        'demons': 3,
        'gc': 3,
        'icp': 3,
        'pse': 3,
        'jhct': 3,
        'igdm': 5
    }

    def __init__(self, metric_name, *args):
        if metric_name.lower() not in self.metrics.keys():
            raise ValueError("Unknown transformation name: " + metric_name)
        # -2 because fixed_image/pointset and moving_image/pointset are to be
        # defined later
        if self.expected_param[metric_name.lower()] - 2 > len(args):
            raise ValueError(
                self.metrics[metric_name.lower()] +
                " transformation requires at least " +
                str(self.expected_param[metric_name.lower()]) +
                " parameters")
        self.partial_metric = lambda fixed_image, moving_image: \
            self.__complete_string(metric_name, fixed_image, moving_image,
                                   *args)

        self.__out_str = None
        self.fixed_image = None
        self.moving_image = None

    @classmethod
    def complete_metric(cls, metric_name, fixed_image, moving_image, *args):
        metric = cls(metric_name, *args)
        metric.complete(fixed_image, moving_image)
        return metric

    @classmethod
    def from_string(cls, string):
        p = re.compile('\w*\[(.*,)+(.+)\]')
        if p.match(string):
            metric_name = string.split('[')[0]
            metric_param = string.split('[')[1].split(']')[0].split(',')
            metric_param = [s.strip() for s in metric_param]
            # 2 or less parameters means either a partial Metric or a wrong
            # format
            if len(metric_param) > 2:
                try:
                    metric = cls.complete_metric(metric_name, *metric_param)
                except ValueError:
                    print("cannot create a complete Metric, trying to create"
                          " a partial Metric")
                    metric = cls(metric_name, *metric_param)
            else:
                metric = cls(metric_name, *metric_param)
        else:
            raise ValueError(string + " has the wrong format, the Metric cannot"
                                      "be created")
        return metric

    def is_complete(self):
        return self.__out_str is not None

    def get_str(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.__out_str

    def get_fixed_image(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.fixed_image


    def get_moving_image(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.moving_image

    def complete(self, fixed_image, moving_image):
        if not os.path.exists(fixed_image):
            raise ValueError(fixed_image + " is not an existing file")
        if not os.path.exists(moving_image):
            raise ValueError(moving_image + " is not an existing file")
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.partial_metric(fixed_image, moving_image)

    def __complete_string(self, metric_name, fixed_image, moving_image, *args):
        self.__out_str = self.metrics[metric_name.lower()] + '[' + \
                       ','.join([fixed_image, moving_image]) + ',' + \
                       ','.join([str(a) for a in args]) + ']'


class Transform(object):
    transform_type = {
        'rigid': 'Rigid',
        'affine': 'Affine',
        'compositeaffine': 'CompositeAffine',
        'similarity': 'Similarity',
        'translation': 'Translation',
        'bspline': 'BSpline',
        'gaussiandisplacementfield': 'GaussianDisplacementField',
        'bsplinedisplacementfield': 'BSplineDisplacementField',
        'timevaryingvelocityfield': 'TimeVaryingVelocityField',
        'timevaryingbsplinevelocityfield': 'TimeVaryingBSplineVelocityField',
        'syn': 'SyN',
        'bsplinesyn': 'BSplineSyN',
        'exponential': 'Exponential',
        'bsplineexponential': 'BSplineExponential'
    }
    # Minimum number of arguments expected for the transformations
    expected_param = {
        'rigid': 1,
        'affine': 1,
        'compositeaffine': 1,
        'similarity': 1,
        'translation': 1,
        'bspline': 2,
        'gaussiandisplacementfield': 3,
        'bsplinedisplacementfield': 3,
        'timevaryingvelocityfield': 6,
        'timevaryingbsplinevelocityfield': 2,
        'syn': 3,
        'bsplinesyn': 3,
        'exponential': 3,
        'bsplineexponential': 3
    }

    def __init__(self, transform_name, gradient_step, *args):
        if transform_name.lower() not in self.transform_type.keys():
            raise ValueError("Unknown transformation name: " + transform_name)
        if self.expected_param[transform_name.lower()] > len(args) + 1:
            raise ValueError(
                self.transform_type[transform_name.lower()] +
                " transformation requires at least " +
                str(self.expected_param[transform_name.lower()]) +
                " parameters")

        self.__out_str = self.transform_type[transform_name.lower()] + '[' + \
            ','.join([str(gradient_step)] + [str(a) for a in args]) + ']'

    @classmethod
    def from_string(cls, string):
        p = re.compile('\w+\[([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(,.+)*)\]$')
        if p.match(string):
            transform_name = string.split('[')[0]
            transform_param = string.split('[')[1].split(']')[0].split(',')
            transform_param = [s.strip() for s in transform_param]
            transform = cls(transform_name, *transform_param)
        else:
            raise ValueError(string + " has the wrong format, the Transform"
                                      " cannot be created")
        return transform

    def get_str(self):
        return self.__out_str


'''--transform Rigid[0.1] --metric MI[$t1brain,$template,1,32,Regular,0.25] \  
        --convergence [1000x500x250x100,1e-6,10] \  
        --shrink-factors 8x4x2x1 \  
        --smoothing-sigmas 3x2x1x0vox \
        
        level_iters=[10000, 1000, 100],
                           sigmas=[3.0, 1.0, 0.0],
                           factors=[4, 2, 1]
'''
def x_str(*args):
    """
    Convert a list of parameters into a string with the format:
    arg1xarg2xargs3 ...
    Parameters
    ----------
    args: list of parameters
        if there is only one parameter which is an Iterable
        (tuple, list, dict, numpy array ....), the output will be a string with
        the content of the Iterable concatenated and separated by 'x' (note that
        for dict-like objects, the keys will be concatenated).


    Returns
    -------

    """
    if len(args) == 1 and isinstance(args[0], Iterable):
        # If there is only one string, returns the string
        if isinstance(args[0], six.string_types):
            return args[0]
        else:
            return 'x'.join([str(a) for a in args[0]])
    return 'x'.join([str(a) for a in args])


class TransformParam(object):
    def __init__(self, transform, metric, convergence, shrink, sigmas):
        if isinstance(transform, Transform):
            self.transform_str = transform.get_str()
        elif isinstance(transform, six.string_types):
            self.transform_str = Transform.from_string(transform).get_str()
        else:
            raise TypeError("transform can either be a Transform object " +
                            "or a string")

        self.convergence = x_str(convergence)
        if not (self.convergence.startswith('[') and
                self.convergence.endswith(']')):
            self.convergence = '[' + self.convergence + ']'
        self.sigmas = x_str(sigmas)
        self.shrink = x_str(shrink)

        self.__out_str = None

        if isinstance(metric, six.string_types):
            self.metric = Metric.from_string(metric)
        elif isinstance(metric, Metric):
            self.metric = metric
            if metric.is_complete():
                self.metric_str = metric.get_str()
        else:
            raise TypeError("metric can either be a Metric object " +
                            "or a string")

        if self.metric.is_complete():
            self.metric_str = metric.get_str()
            self.__complete_string()

    @classmethod
    def complete_transform(cls, fixed_image, moving_image, transform,
                           metric, convergence, shrink, sigmas):
        if not metric.is_complete():
            metric.complete(fixed_image, moving_image)
        return TransformParam(transform, metric, convergence, shrink,
                              sigmas)

    def is_complete(self):
        return self.__out_str is not None

    def get_str(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.__out_str

    def get_fixed_image(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.metric.fixed_image

    def get_moving_image(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.metric.moving_image

    def complete(self, fixed_image, moving_image):
        self.metric.complete(fixed_image, moving_image)
        self.metric_str = self.metric.get_str()
        self.__complete_string()

    def __complete_string(self):
        self.__out_str = ' '.join(['--transform',
                                   self.transform_str,
                                   '--metric',
                                   self.metric_str,
                                   '--convergence',
                                   self.convergence,
                                   '--shrink-factors',
                                   self.shrink,
                                   '--smoothing-sigmas',
                                   self.sigmas])


class AntsRegistration(object):
    def __init__(self,
                 transforms,
                 moving_image=None,
                 fixed_image=None,
                 output='transform',
                 dim=3,
                 interp='Linear',
                 winsorize='[0.005,0.995]',
                 use_histo_matching=1,
                 init_transform=None,
                 masks=None,
                 use_float=0,
                 collapse_out_transforms='1'):
        if isinstance(transforms, TransformParam):
            self.transform_lst = [transforms]
        elif isinstance(transforms, six.string_types):
            self.transform_lst = [Transform.from_string(transforms)]
        elif isinstance(transforms, list):
            self.transform_lst = []
            for tr in transforms:
                if isinstance(tr, TransformParam):
                    self.transform_lst.append(tr)
                elif isinstance(tr, six.string_types):
                    self.transform_lst.append(Transform.from_string(tr))
                else:
                    raise TypeError(
                        'transforms can either be a Transform object,'
                        'a string or a list of Transform or strings')
        else:
            raise TypeError('transforms can either be a Transform object,'
                            'a string or a list of Transform or strings')

        """ Could also add the possibility to use the same parameters for
        several transformations. Like defining the interrations once and if it
        is not provided for the other transformations the first one will be 
        repeated.
        """

        self.transforms_str = ' '.join(
            [tr.get_str() for tr in self.transform_lst])

        self.output_str = None
        if output is not None and output != '':
            self.out_pref = str(output)
            self.output_str = ' '.join(["--output",
                                        '[' + self.out_pref + ','
                                        + self.out_pref
                                        + '_Warped.nii.gz]'])
            self.output_mat_file = (self.out_pref + '.mat')
        self.dim_str = ' '.join(['--dimensionality', str(dim)])
        self.collapse_out_transforms_str = ' '.join(
            ["--collapse-output-transforms", str(collapse_out_transforms)])
        self.interp_str = ' '.join(["--interpolation", str(interp)])
        self.winsor_str = ' '.join(["--winsorize-image-intensities",
                                    str(winsorize)])
        self.histo_str = ' '.join(["--use-histogram-matching ",
                                   str(use_histo_matching)])
        self.use_float_str = ' '.join(['--float', str(use_float)])

        self.init_transform = init_transform
        self.init_transform_str = None

        self.masks_str = ''
        if masks is not None:
            if re.match('^\[\w+,\w+\]$', masks):
                fixed = masks.split('[')[1].split(',')[0]
                moving = masks.split(']')[0].split(',')[1]
                if not os.path.exists(fixed):
                    raise ValueError(fixed + ' does not exist')
                if not os.path.exists(moving):
                    raise ValueError(moving + ' does not exist')
            else:
                if not os.path.exists(masks):
                    raise ValueError(masks + ' does not exist')
            self.masks_str = '--masks ' + masks

        self.moving_image = None
        self.fixed_image = None
        if moving_image is None and fixed_image is None:
            if all([tr.is_complete() for tr in self.transform_lst]):
                self.set_moving_image(self.transform_lst[0].get_moving_image())
                self.set_fixed_image(self.transform_lst[0].get_fixed_image())
        else:
            if moving_image is not None:
                self.set_moving_image(moving_image)
            if fixed_image is not None:
                self.set_fixed_image(fixed_image)

    def is_complete(self):
        return self.__out_str is not None

    def get_str(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.__out_str

    def get_fixed_image(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.metric.fixed_image

    def get_moving_image(self):
        if not self.is_complete():
            raise ValueError("The Metric has not been completed with the "
                             "fixed and moving image paths")
        else:
            return self.metric.moving_image

    def set_fixed_image(self, fixed_image):
        # Overwrites the fixed images set for all transformations
        if os.path.exists(fixed_image):
            self.fixed_image = fixed_image
        else:
            raise ValueError(fixed_image + " does not exist")

        if self.moving_image is not None:
            self.complete(self.fixed_image, self.moving_image)

    def set_moving_image(self, moving_image):
        # Overwrites the moving images set for all transformations
        if os.path.exists(moving_image):
            self.moving_image = moving_image
        else:
            raise ValueError(moving_image + " does not exist")

        if self.fixed_image is not None:
            self.complete(self.fixed_image, self.moving_image)

    def complete(self, fixed_image, moving_image):
        if self.init_transform is not None:
            if re.match("^([012])$", str(self.init_transform)):
                self.init_transform_str = '[' + self.moving_image + ',' + \
                                          self.fixed_image + \
                                          str(self.init_transform) + ']'
            elif isinstance(self.init_transform, six.string_types):
                if os.path.exists(self.init_transform):
                    self.init_transform_str = self.init_transform
                elif re.match("^[.+,(0|1)]$", self.init_transform):
                    init_tr_path = self.init_transform.split('[')[1].split(
                            ']')[0].split(',')[0]
                    if os.path.exists(init_tr_path):
                        self.init_transform_str = self.init_transform
                    else:
                        raise ValueError(init_tr_path + "does not exist")
                else:
                    raise ValueError(self.init_transform + "does not have the "
                                                           "right format")
            else:
                raise ValueError(str(self.init_transform) + "does not have the"
                                                            " right format")

            self.init_transform_str = ' '.join(["--initial-moving-transform",
                                                self.init_transform_str])
        self.moving_image = moving_image
        self.fixed_image = fixed_image

        if self.output_str is None:
            base = os.path.basename(self.moving_image)
            base = base.split('.')[0]
            self.out_pref = 'transform_' + base
            self.output_str = ' '.join(["--output",
                                        '[' + self.out_pref + ','
                                        + self.out_pref
                                        + '_Warped.nii.gz]'])

        self.output_mat_file = glob.glob(self.out_pref + "*.mat")
        for tr in self.transform_lst:
            tr.complete(self.fixed_image, self.moving_image)
        self.__complete_string()

    def __complete_string(self):
        # filter(None ... will remove the strings with None or '' from the list
        self.__out_str = ' '.join(filter(None, [
            'antsRegistration',
            self.transforms_str,
            self.output_str,
            self.dim_str,
            self.collapse_out_transforms_str,
            self.interp_str,
            self.winsor_str,
            self.histo_str,
            self.use_float_str,
            self.init_transform_str,
            self.masks_str
        ]))

    def get_transform_norm(self):
        print(self.__out_str)

    def run(self):
        # run ants reg
        # set the output filesnames
        # calculate norm
        if self.is_complete():
            try:
                retcode = subprocess.check_output(self.get_str(),shell=True)
            except Exception as e:
                raise Exception(
                    '[!] ANTS registration did not complete successfully.'
                    '\n\nError details:\n{0}\n'.format(e))
            return retcode
        else:
            raise ValueError(str(self) + ' is not complete, please make sure '
                                         'to set the fixed and moving images')


def ants_registration(transforms,
                      moving_image=None,
                      fixed_image=None,
                      output=None,
                      dim=3,
                      interp='Linear',
                      winsorize='[0.005,0.995]',
                      use_histo_matching=1,
                      init_transform=None,
                      masks=None,
                      use_float=0,
                      collapse_out_transforms='0'):
    if isinstance(transforms, TransformParam):
        transform_lst = [transforms]
    elif isinstance(transforms, six.string_types):
        transform_lst = [Transform.from_string(transforms)]
    elif isinstance(transforms, list):
        transform_lst = []
        print(str(transform_lst))
        print(type(transform_lst))
        for tr in transforms:
            if isinstance(tr, TransformParam):
                transform_lst.append(tr)
            elif isinstance(tr, six.string_types):
                transform_lst.append(Transform.from_string(tr))
            else:
                raise TypeError('transforms can either be a Transform object,'
                                ' a string or a list of Transform or strings')
    else:
        raise TypeError('transforms can either be a Transform object,'
                        ' a string or a list of Transform or strings')

    transforms_str = ' '.join([tr.get_str() for tr in transform_lst])

    out_pref = str(output)
    output_str = ' '.join(["--output", '[' + out_pref + ',' + out_pref
                           + '_Warped.nii.gz]'])
    dim_str = ' '.join(['--dimensionality', str(dim)])
    collapse_out_transforms_str = ' '.join(["--collapse-output-transforms",
                                           str(collapse_out_transforms)])
    interp_str = ' '.join(["--interpolation", str(interp)])
    winsor_str = ' '.join(["--winsorize-image-intensities", str(winsorize)])
    histo_str = ' '.join(["--use-histogram-matching ", str(use_histo_matching)])
    use_float_str = ' '.join(['--float', str(use_float)])

    """ Could be modified to allow the modification of either or both moving
    and fixed image
    """
    if moving_image is None and fixed_image is None:
        if all([tr.is_complete() for tr in transform_lst]):
            mov_img = transform_lst[0].get_moving_image()
            fix_img = transform_lst[0].get_fixed_image()
        else:
            raise ValueError("moving and fixed images has to me defined"
                             "either in the Transform/Metric objects or"
                             "in the moving_image and fixed_image parameters")
    elif moving_image is not None and fixed_image is not None:
        mov_img = moving_image
        fix_img = fixed_image
        for tr in transform_lst:
            tr.complete(fix_img, mov_img)
    else:
        raise ValueError("moving and fixed images has to me defined"
                         "either in the Transform/Metric objects or"
                         "in the moving_image and fixed_image parameters")

    if init_transform is not None:
        if re.match("^(0|1|2)$", str(init_transform)):
            init_transform_str = '[' + mov_img + ',' + fix_img + \
                                 str(init_transform) + ']'
        elif isinstance(init_transform, six.string_types):
            if os.path.exists(init_transform):
                init_transform_str = init_transform
            elif re.match("^[.+,(0|1)]$", init_transform):
                init_tr_path = init_transform.split('[')[1].split(
                        ']')[0].split(',')[0]
                if os.path.exists(init_tr_path):
                    init_transform_str = init_transform
                else:
                    raise ValueError(init_tr_path + " does not exist")
            else:
                raise ValueError(init_transform + " does not have the right"
                                                  "format")
        else:
            raise ValueError(str(init_transform) + " does not have the right"
                                                   "format")

        init_transform_str = ' '.join(["--initial-moving-transform",
                                      init_transform_str])
    else:
        init_transform_str = ''

    masks_str = ''
    if masks is not None:
        if re.match('￿^\[\w+,\w+\]$'):
            fixed = masks.split('[')[1].split(',')[0]
            moving = masks.split(']')[0].split(',')[1]
            if not os.path.exists(fixed):
                raise ValueError(fixed + ' does not exist')
            if not os.path.exists(moving):
                raise ValueError(moving + ' does not exist')
        else:
            if not os.path.exists(masks):
                raise ValueError(masks + ' does not exist')
        masks_str = '--masks ' + masks

    command = ' '.join(filter(None, [
        'antsRegistration',
        transforms_str,
        output_str,
        dim_str,
        collapse_out_transforms_str,
        interp_str,
        winsor_str,
        histo_str,
        use_float_str,
        init_transform_str,
        masks_str
    ]))

    return command


# fixed_path = '/Users/cf27246/test/MNI152_T1_3mm_brain.nii.gz'
# moving_path = '/Users/cf27246/test/test_fmri_mean.nii.gz'
#
# m0 = Metric('CC', 1, 4)
# print(m0.is_complete())
# m0.complete(fixed_path, moving_path)
# print(m0.is_complete())
# print(m0.get_str())
#
# m1 = Metric.complete_metric('CC', fixed_path, moving_path, 1, 4)
# print(m1.is_complete())
# print(m1.get_str())
#
# m2 = Metric.from_string('CC[' + ','.join([fixed_path, moving_path, '1, 4]']))
# print(m2.is_complete())
# print(m2.get_str())
#
# m3 = Metric.from_string('CC[1, 4]')
# print(m3.is_complete())
# m3.complete(fixed_path, moving_path)
# print(m3.is_complete())
# print(m3.get_str())
#
#
# t1 = Transform('Rigid', 0.1)
# print(t1.get_str())
#
# t2 = Transform.from_string('Rigid[0.1]')
# print(t2.get_str())
#
# fixed_path = '/Users/cf27246/test/MNI152_T1_3mm_brain.nii.gz'
# moving_path = '/Users/cf27246/test/test_fmri_mean.nii.gz'
#
# m0 = Metric('CC', 1, 4)
#
# m1 = Metric.complete_metric('CC', fixed_path, moving_path, 1, 4)
#
# m2 = Metric.from_string('CC[' + ','.join([fixed_path, moving_path, '1, 4]']))
#
# m3 = Metric.from_string('CC[1, 4]')
#
# tp0 = TransformParam(t1, m0, '[1000x500x250x100,1e-08,10]',
#                      [8,4,2,1], [3,2,1,0])
# print(tp0.is_complete())
# tp0.complete(fixed_path, moving_path)
# print(tp0.is_complete())
# print(m0.is_complete())
# print(tp0.get_str())
#
# tp1 = TransformParam(t1, m1, '[1000x500x250x100,1e-08,10]',
#                      [8,4,2,1], [3,2,1,0])
# print(tp1.is_complete())
# print(tp1.get_str())
#
# tp2 = TransformParam(t2, m2, '[1000x500x250x100,1e-08,10]',
#                      [8,4,2,1], [3,2,1,0])
# print(tp2.is_complete())
# print(tp2.get_str())
#
# tp3 = TransformParam(t2, m3, '[1000x500x250x100,1e-08,10]',
#                      [8, 4, 2, 1], [3 ,2 ,1 ,0])
# print(tp3.is_complete())
# tp3.complete(fixed_path, moving_path)
# print(tp3.is_complete())
# print(tp3.get_str())
#
#
# im1 = '/Users/cf27246/test/ants_reg/test_linear/median_sub-0027251_ses-1_' \
#       'task-msit_run-1_bold.nii.gz'
# template = '/Users/cf27246/test/ants_reg/test_linear/test_median_fmri.nii.gz'
# m11 = Metric.complete_metric('CC', template, im1, 1, 4)
# tp11 = TransformParam(Transform('Rigid', 0.1),
#                       m11, '[100,1e-06,10]',
#                       [1], [0])
# tp22 = TransformParam(Transform('Affine', 0.1),
#                       m11, '[100,1e-06,10]',
#                       [1], [0])
# print(tp11.is_complete())
# print(tp11.get_str())
# print(tp22.is_complete())
# print(tp22.get_str())
# regcmd = ants_registration([tp11, tp22])
#
# titi_reg = AntsRegistration([tp11, tp22], template, template)
# titi_reg.run()
#
# os.system("echo '%s' | pbcopy" % titi_reg.get_str())
#
# os.system("echo '%s' | pbcopy" % regcmd)
# subprocess.check_output(regcmd)
# # with open(command_file, 'wt') as f:
# #     f.write(' '.join(regcmd))
#
# try:
#     retcode = subprocess.check_output(regcmd)
# except Exception as e:
#     raise Exception('[!] ANTS registration did not complete successfully.'
#                     '\n\nError details:\n{0}\n{1}\n'.format(e, e.output))
#
#
# def loop_ants_reg(ants_reg_list,
#                   thread_pool=None):
#     if isinstance(thread_pool, int):
#         pool = ThreadPool(thread_pool)
#     else:
#         pool = thread_pool
#
#     pool.map(lambda ants_reg: ants_reg.run(), ants_reg_list)
#
#     if isinstance(thread_pool, int):
#         pool.close()
#         pool.join()



    # --initial - moving - transform
    # initialTransform
    # [initialTransform, < useInverse >]
    # [fixedImage, movingImage, initializationFeature]




    # @classmethod
    # def from_json(cls, book_as_json: str) -> 'Book':
    #     book = json.loads(book_as_json)
    #     return cls(title=book['title'], author=book['author'],
    #                pages=book['pages'])




# def ants_reg(moving_image,
#              fixed_image,
#              transforms,
#              output,
#              interpolation,
#              init_):
#
#     antsRegistration --dimensionality 3 --float 0 \
#         --output [$thisfolder/pennTemplate_to_${sub}_,$thisfolder/pennTemplate_to_${sub}_Warped.nii.gz] \
#         --interpolation Linear \
#         --winsorize-image-intensities [0.005,0.995] \
#         --use-histogram-matching 0 \
#         --initial-moving-transform [$t1brain,$template,1] \
#         --transform Rigid[0.1] \
#         --metric MI[$t1brain,$template,1,32,Regular,0.25] \
#         --convergence [1000x500x250x100,1e-6,10] \
#         --shrink-factors 8x4x2x1 \
#         --smoothing-sigmas 3x2x1x0vox \
#         --transform Affine[0.1] \
#         --metric MI[$t1brain,$template,1,32,Regular,0.25] \
#         --convergence [1000x500x250x100,1e-6,10] \
#         --shrink-factors 8x4x2x1 \
#         --smoothing-sigmas 3x2x1x0vox \
#         --transform SyN[0.1,3,0] \
#         --metric CC[$t1brain,$template,1,4] \
#         --convergence [100x70x50x20,1e-6,10] \
#         --shrink-factors 8x4x2x1 \
#         --smoothing-sigmas 3x2x1x0vox \
#         -x $brainlesionmask
#
# import timeit
# print(timeit.timeit("test()", setup="from __main__ import test", number=1))
