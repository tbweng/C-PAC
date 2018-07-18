

def mask_summarize_time_course(functional_file_path, mask_file_path, output_file_path, method="DetrendNormMean",
                               mask_vol_index=None, mask_label=None, num_pcs=1):
    """
    Calculates summary time course for voxels specified by a mask. Methods for summarizing the the time courses include
    mean and principle component analysis.

    :param functional_file_path: path to nifti file corresponding to the functional data that will be used in the mean
            calculation.
    :param mask_file_path: a nifti file indicating the voxels that should be included in the summary. The mask may
            include multiple regions and multiple volumes, which can be specified using the 'mask_vol_index' and
            'mask_label' parameters. Default is to calculate a separate summary for each volume of the mask file
            and to combine all non-zero regions of each volume into the same summary. The orientation, voxel sizes, and
            'space' of the mask should match those of the functional data.
    :param output_file_path: full path and name of the output TSV
    :param method: the method used for calculating the summary. suitable values are:
            'Mean' - the average of voxel indices
            'NormMean' - z-score normalize voxels prior to calculating summary
            'DetrendNormMean' - detrend (degree=2) and normalize voxels prior to calculating summary
            'PCA' - summarize data as top num_pcs from a principle components analysis (i.e. compcor)
            default is 'DetrendNormMean'
    :param mask_vol_index: (int) if the mask contains multiple volumes, index of the volume to use. default behavior is
            to calculate a separate summary for each volume, which are output to separate columns of the output TSV
    :param mask_label: (list of tuples) if a mask contains multiple labelled regions, the desired regions can be
            selected by providing a list containing a tuple for each volume to be considered, if more than one label is
            provided, the union of the specified regions will be used
    :param num_pcs: If PCA is chosen, the number of the largest PCs that should be returned. Default is the first.
    :return: name of the TSV file containing the output.
    """

    import os
    import nibabel as nb
    import numpy as np

    # first check inputs to make sure that they are OK
    if not functional_file_path or \
            (not functional_file_path.endswith(".nii.gz") and not functional_file_path.endswith(".nii")) \
            or not os.path.isfile(functional_file_path):
        raise ValueError("Invalid value for functional_data ({}). Should be the filename to an input nifti "
                         "file".format(functional_file_path))

    if not mask_file_path or (not mask_file_path.endswith(".nii.gz") and not mask_file_path.endswith(".nii")) or \
            not os.path.isfile(mask_file_path):
        raise ValueError("Invalid value for mask ({}). Should be the path to an existing nifti "
                         "file".format(mask_file_path))

    if not output_file_path:
        raise ValueError("Invalid value for output_file_path ({}).".format(output_file_path))

    if method and method not in ['Mean', 'NormMean', 'DetrendNormMean', 'PCA']:
        raise ValueError("Invalid value for method ({}). Should be one of ['Mean', 'NormMean', 'DetrendNormMean',"
                         " 'PCA']".format(method))

    if mask_vol_index and not (isinstance(mask_vol_index, (int, long)) and mask_vol_index > 0):
        raise ValueError("Invalid value for mask_vol_index ({}). Should be an integer > 0".format(mask_vol_index))

    if mask_label:
        if not (isinstance(mask_label, list)):
            raise ValueError("Invalid type for mask_label ({}). Should be a list.".format(type(mask_label), mask_label))

        for mask_label_index, mask_label_val in enumerate(mask_label):
            if not isinstance(mask_label_val, (int, long)) and not isinstance(mask_label_val, tuple) \
                    and not isinstance(mask_label_val, list):
                raise ValueError("Invalid type ({0}) for mask label #{1}: {2}, should be integer, "
                                 "list, or tuple".format(type(mask_label_val), mask_label_index, mask_label_val))

    if num_pcs and not (isinstance(num_pcs, (int, long)) and num_pcs > 0):
        raise ValueError("Invalid value for num_pcs ({0}). Should be an integer > 0".format(num_pcs))

    functional_image = nb.load(functional_file_path)

    if len(functional_image.shape) < 4 or functional_image.shape[3] == 1:
        raise ValueError("Expecting 4D file nifti file, but {0} has shape {1}.".format(functional_file_path,
                                                                                       functional_image.shape))

    functional_image_data = (functional_image.get_data()).reshape((np.prod(functional_image.shape[0:3]),
                                                                   functional_image.shape[3]))

    time_course_length = functional_image.shape[3]

    print("Mask image data comes from {0}".format(mask_file_path))

    # read in the mask data
    mask_image = nb.load(mask_file_path)

    if mask_image.shape[0:3] != functional_image.shape[0:3] or \
            not np.allclose(mask_image.affine, functional_image.affine):
        raise ValueError("Mask ({0}) and functional image ({1}) must be in the same space. Please check the header "
                         "and verify that the shape and transform are the same.".format(mask_file_path,
                                                                                        functional_file_path))

    if len(mask_image.shape) > 3:

        mask_image_data = mask_image.get_data().reshape((np.prod(mask_image.shape[0:3]), mask_image.shape[3]))

        if mask_vol_index and mask_vol_index >= mask_image.shape[3]:
            raise ValueError("Requested mask volume {0}, from file that only contains "
                             "volumes 0 ... {1}".format(mask_vol_index, mask_vol_index - 1))

        mask_image_data = mask_image_data[:, [mask_vol_index]]

    else:

        mask_image_data = mask_image.get_data().reshape((np.prod(mask_image.shape[0:3]), 1))

        if mask_vol_index and mask_vol_index != 0:
            print("Requested mask volume {0}, from file that only contains a single volume (0), "
                  "ignoring.".format(mask_vol_index))

    if mask_label and len(mask_label) > 1 and len(mask_label) != mask_image_data.shape[1]:
        raise ValueError("Length of mask labels {0} should either match the number of mask volumes {1}, "
                         "or should be 1".format(len(mask_label), mask_image_data.shape[1]))

    number_of_mask_volumes = mask_image_data.shape[1]

    # the calculated roi summaries will be held in a list of lists
    time_course_summaries = []

    for mask_index in range(0, mask_image_data.shape[1]):

        # extract the time series for the intended voxels
        if not mask_label:
            voxel_indices = np.where(mask_image_data[:, mask_index] > 0)[0].tolist()

            print("{0} nonzero positive voxels in mask {1}".format(len(voxel_indices), mask_file_path))

        else:
            # if we only have one set of mask labels, but many mask volumes, we apply the same labels to each volume,
            # otherwise we use a different set of mask labels for each volume
            mask_label_this_mask_vol = []
            if len(mask_label) == 1:
                mask_label_this_mask_vol = mask_label[0]
            else:
                mask_label_this_mask_vol = mask_label[mask_index]

            if isinstance(mask_label_this_mask_vol, int):
                mask_label_this_mask_vol = [mask_label_this_mask_vol]

            # calculate the union between the different masks
            voxel_indices = []
            for mask_label_val in mask_label_this_mask_vol:
                voxel_indices += np.where(mask_image_data[:, mask_index] == mask_label_val)[0].tolist()

            print("{0} voxels in mask {1} with labels {2}".format(len(voxel_indices), mask_file_path,
                                                                  mask_label_this_mask_vol))

        # make sure that the voxel indices are unique
        voxel_indices = np.unique(voxel_indices)

        if len(voxel_indices) == 0:
            raise ValueError("Time series extraction failed, no voxels in mask {0} match label(s) {1}".format(
                mask_file_path, mask_label))

        time_courses = functional_image_data[voxel_indices, :]

        # exclude time courses with zero variance to avoid wasting our time on meaningless data, and to avoid
        # NaNs
        time_courses = time_courses[np.isclose(time_courses.var(1), 0.0) == False, :]

        if time_courses.shape[0] == 0:
            raise ValueError("None of the {0} in-mask voxels have non-zero variance time"
                             " courses.".format(len(voxel_indices)))

        print("{0} voxels survived variance filter".format(time_courses.shape[0]))

        # Make sure we have a 2D array, even if it only contains a single time course
        if len(time_courses.shape) == 1:
            time_courses = time_courses.reshape((time_courses.shape[0], 1))

        if method in ["PCA"]:
            # compcor begins with linearly detrending the columns of the matrix
            def linear_detrend_columns(image_array_2d):
                """
                perform quadratic detrending on each row of a 2D numpy array representing a functional image

                :param image_array_2d: 2D numpy array containing functional data to be processed
                :return: 2D numpy array of residuals
                """
                column_len = image_array_2d.shape[0]
                polynomial_design_matrix = np.array([range(0, column_len), [1] * column_len])
                polynomial_coefficients = np.polyfit(range(0, column_len), image_array_2d, deg=1)
                return image_array_2d - polynomial_coefficients.transpose().dot(polynomial_design_matrix).transpose()

            time_courses = linear_detrend_columns(time_courses)

        # normalize data as requested
        if method in ["PCA", "DetrendNormMean"]:
            def quadratic_detrend_rows(image_array_2d):
                """
                perform quadratic detrending on each row of a 2D numpy array representing a functional image

                :param image_array_2d: 2D numpy array containing functional data to be processed
                :return: 2D numpy array of residuals
                """

                print("2D image shape {0} {1}".format(image_array_2d.shape, len(image_array_2d.shape)))

                row_len = image_array_2d.shape[1]
                polynomial_design_matrix = np.array([[x * x for x in range(0, row_len)], range(0, row_len),
                                                     [1] * row_len])
                polynomial_coefficients = np.polyfit(range(0, row_len), image_array_2d.transpose(), deg=2)
                return image_array_2d - polynomial_coefficients.transpose().dot(polynomial_design_matrix)

            time_courses = quadratic_detrend_rows(time_courses)

        if method in ["PCA", "DetrendNormMean", "NormMean"]:
            time_courses = time_courses - np.tile(time_courses.mean(1).reshape(time_courses.shape[0], 1),
                                                  (1, time_courses.shape[1]))
            time_courses = time_courses / np.tile(time_courses.std(1).reshape(time_courses.shape[0], 1),
                                                  (1, time_courses.shape[1]))

        # now summarise
        if method in ["DetrendNormMean", "NormMean", "Mean"]:

            time_course_summaries.append(time_courses.mean(0))

        elif method in ["PCA"]:

            [u, s, v] = np.linalg.svd(time_courses, full_matrices=False)
            time_course_summaries.append(v[0:num_pcs, :])

    if len(time_course_summaries) != number_of_mask_volumes:
        raise ValueError("Expected {0} summaries, one for mask volume, "
                         "but received {1}".format(number_of_mask_volumes, len(time_course_summaries)))

    output_file_path = os.path.join(os.getcwd(), output_file_path)
    with open(output_file_path, "w") as ofd:

        for time_point in range(0, time_course_length):

            row_values = []

            if time_point == 0:

                # write out the header information
                ofd.write("# CPAC version {0}\n".format("1.010"))
                ofd.write("# Time courses extracted from {0} using mask {1} and "
                          "method {2}\n".format(functional_file_path, mask_file_path, method))

                for time_course_summary_index, time_course_summary in enumerate(time_course_summaries):
                    if method in ["PCA"]:
                        if time_course_summary.shape[0] != num_pcs:
                            raise ValueError("Time course summary {0} expected {1} pcs, but received {2}".format(
                                time_course_summary_index, num_pcs, time_course_summary.shape[0]))
                        row_values += ["mask#{0}_{1}#{2}".format(time_course_summary_index, method, pc)
                                       for pc in range(0, num_pcs)]
                    else:
                        row_values += ["mask#{0}_{1}".format(time_course_summary_index, method)]

                ofd.write("#" + "\t".join(row_values) + "\n")

                row_values = []

            for time_course_summary_index, time_course_summary in enumerate(time_course_summaries):
                if method in ["PCA"]:
                    row_values += ["{0}".format(time_course_summary[pc, time_point]) for pc in range(0, num_pcs)]
                else:
                    row_values += ["{0}".format(time_course_summary[time_point])]

            ofd.write("\t".join(row_values) + "\n")

    return output_file_path


def insert_mask_summarize_time_course_node(workflow, functional_source, mask_source, mask_vol_index=None,
                                           mask_label=None, summarization_method="DetrendNormMean", num_pcs=1,
                                           node_name='wm_summarize'):

    """

    Insert a nipype node that extracts a time course from a mask region into a workflow.

    :param workflow: the workflow to connect the node to
    :param functional_source: the 4D functional file to extract the time series from, this can either be a path to a
         file or a (node, resource) tuple corresponding to a nipype node that produces the file
    :param mask_source: mask file that defines the region to extract the time series from, this can either be a path to
         a file or a (node, resource) tuple corresponding to a nipype node that produces the file. All non-zero voxels
         in the first volume in the mask file will be used to define the region unless mask_vol_index and mask_label are
         defined
    :param mask_vol_index: the index of the volume in the mask_source to be used to define the region, defaults to 0
    :param mask_label: integer or list of integer corresponding to the value in the mask file that defines the region
          to extract a time series for, defaults to all non-zero values
    :param summarization_method: suitable values are:
            'Mean' - the average of voxel indices
            'NormMean' - z-score normalize voxels prior to calculating summary
            'DetrendNormMean' - detrend (degree=2) and normalize voxels prior to calculating summary
            'PCA' - summarize data as top num_pcs from a principle components analysis (i.e. compcor)
    :param num_pcs: number of PCs to extract if using PCA, defaults to 1
    :param node_name: name to be given to the nipype node, will also be used to construct output filename
    :return: (node, resource) tuple pointing to the generated resource file
    """

    import nipype.pipeline.engine as pe
    import os
    import nipype.interfaces.utility as util
    
    mask_summarize_time_course_node = pe.Node(util.Function(input_names=['functional_file_path',
                                                                         'mask_file_path',
                                                                         'output_file_path',
                                                                         'summary_method',
                                                                         'mask_vol_index',
                                                                         'mask_label',
                                                                         'num_pcs'],
                                                            output_names=['regressor_file_path'],
                                                            function=mask_summarize_time_course),
                                              name='summarise_regressor_{}'.format(node_name))

    if isinstance(functional_source, tuple):
        if isinstance(functional_source[0], pe.Node) and isinstance(functional_source[1], str):
            if not workflow:
                raise ValueError("Workflow is required to connect functional_source to the time series summarize node. "
                                 "Received None")
            workflow.connect(functional_source[0], functional_source[1], mask_summarize_time_course_node,
                             'functional_file_path')

    elif os.path.isfile(functional_source):
        mask_summarize_time_course_node.inputs.functional_file_path = mask_source

    if isinstance(mask_source, tuple):
        if isinstance(mask_source[0], pe.Node) and isinstance(mask_source[1], str):
            if not workflow:
                raise ValueError("Workflow is required to connect mask_source to the time series summarize node. "
                                 "Received None")
            workflow.connect(mask_source[0], mask_source[1], mask_summarize_time_course_node, 'mask_file_path')

    elif os.path.isfile(mask_source):
        mask_summarize_time_course_node.inputs.mask_file_path = mask_source

    else:
        raise ValueError("Expected existing file or (source node, string) pair for mask_source, but "
                         "received {0}".format(mask_source))

    mask_summarize_time_course_node.interface.estimated_memory_gb = 1.0

    mask_summarize_time_course_node.inputs.method = 'DetrendNormMean'
    if summarization_method:
        if summarization_method in ['DetrendNormMean', 'NormMean', 'Mean', 'PCA']:
            mask_summarize_time_course_node.inputs.method = summarization_method
        else:
            raise ValueError(
                'Invalid summarization method specified {0}. Should be one of {1}'.format(summarization_method,
                                                                                          ", ".join(['DetrendNormMean',
                                                                                                     'NormMean', 'Mean',
                                                                                                     'PCA'])))
    mask_summarize_time_course_node.inputs.num_pcs = 1
    if num_pcs:
        mask_summarize_time_course_node.inputs.num_pcs = int(num_pcs)

    mask_summarize_time_course_node.inputs.mask_vol_index = 0
    if mask_vol_index:
        mask_summarize_time_course_node.inputs.mask_vol_index = int(mask_vol_index)

    mask_summarize_time_course_node.inputs.mask_label = None
    if mask_label:
        mask_summarize_time_course_node.inputs.mask_label = mask_label

    variant_key = summarization_method
    if summarization_method is "PCA":
        variant_key += "nPCs{0}".format(num_pcs)

    mask_summarize_time_course_node.inputs.output_file_path = \
        'variant-{0}_{1}.tsv'.format(variant_key, node_name)

    return mask_summarize_time_course_node, 'regressor_file_path'
