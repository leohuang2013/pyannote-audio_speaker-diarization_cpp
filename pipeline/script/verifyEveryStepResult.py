import os
import sys
import numpy as np

#checkList = ['batch_waveform','segmentations','batch_masks', 'wav_lens', 'signals', 'masks',
checkList = ['batch_masks', 'segmentations', 'wav_lens', 'signals', 'masks',
        'imasks', 'binarize_score', 'on', 'same_as', 'samples', 'well_defined_idx', 
        'initial_state', 'binarized_segmentations', 
        'binary_ndarray', 'final_wav_lens', 
        'clean_segmentations', 'trimmed', 'sum_trimmed', 'count_data',
        'count', 'embeddings','filtered_embeddings',
        'norm_embeddings', 'clusters', 'dist', 'clusterRes', 
        'soft_clusters', 'hard_clusters', 'clustered_segmentations',
        'aggregated_output', 'aggregated_mask', 'overlapping_chunk_count',
        'scores_in_aggregate', 'masks_in_aggregate',
        'to_diarization_activations', 'cropped_activations','cropped_count', 
        'sorted_speakers', 'discrete_diarization' ]

def deleteSingle( item, source ):
        more = True
        num = 0
        while more:
            fn = f'/tmp/{source}_{item}{num}.txt'
            if os.path.exists( fn ):
                print( f'Delete file: {fn}' )
                os.remove( fn )
                num += 1
            else:
                more = False

        fn = f'/tmp/{source}_{item}.txt'
        if os.path.exists( fn ):
            print( f'Delete file: {fn}' )
            os.remove( fn )

def deleteResultFiles():
    for item in checkList:
        deleteSingle( item, 'cpp' )
        deleteSingle( item, 'py' )

def checkIfSame( fn_cpp, fn_py ):
    f_cpp = open( fn_cpp )
    f_py = open( fn_py )
    if f_cpp.read() == f_py.read():
        same = True
    else:
        same = False
    f_cpp.close()
    f_py.close()
    return same

def requireSameFileContent( item ):
    more = True
    num = 0
    while more:
        fn_cpp = f'/tmp/cpp_{item}{num}.txt'
        fn_py = f'/tmp/py_{item}{num}.txt'
        if os.path.exists( fn_cpp ) and os.path.exists( fn_py ):
            if checkIfSame( fn_cpp, fn_py ):
                print( f"Checking passed for {item}{num}" )
            else:
                print( f"Difference is detected for {item}{num}" )
            num += 1
        else:
            more = False

    fn_cpp = f'/tmp/cpp_{item}.txt'
    fn_py = f'/tmp/py_{item}.txt'
    if os.path.exists( fn_cpp ) and os.path.exists( fn_py ):
        if checkIfSame( fn_cpp, fn_py ):
            print( f"Checking passed for {item}" )
        else:
            print( f"Difference is detected for {item}" )

def local_check( cpp_arr, py_arr ):
    import math
    for i in range( len( cpp_arr )):
        if cpp_arr[i] == py_arr[i]:
            continue
        elif math.isnan( cpp_arr[i] ) and math.isnan( py_arr[i] ):
            continue
        else:
            if math.isnan( cpp_arr[i] ) or math.isnan( py_arr[i] ):
                print( 'nan is different' )
            else:
                print( f'[{i}]: float different: ' + str( cpp_arr[i] - py_arr[i]) )


def checkCloseEnough( fn_cpp, fn_py ):
    f_cpp = open( fn_cpp )
    f_py = open( fn_py )
    cpp_content = f_cpp.read()
    py_content = f_py.read()
    py_content = py_content.replace( '\n', ',' )
    cpp_content = cpp_content.replace( '\n', ',' )
    pyarr = py_content.split( ',' )
    cpparr = cpp_content.split( ',' )
    py_arr = []
    cpp_arr = []
    for a in pyarr:
        if a == '' or a == ' ':
            continue
        py_arr.append( float( a ))
    for a in cpparr:
        if a == '' or a == ' ':
            continue
        cpp_arr.append( float( a ))
    same = True
    #if 'soft_clusters' in fn_cpp:
    #    local_check( cpp_arr, py_arr )
    if len( py_arr ) != len( cpp_arr ):
        print( f'size is different, py: {len(py_arr)}, cpp: {len(cpp_arr)}' )
        return False
    else:
        py_nparr = np.array( py_arr )
        cpp_nparr = np.array( cpp_arr )
        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    py_nparr,
                    cpp_nparr,
                    rtol=0.001,
                    atol=0.0001,
                    verbose = True)
            print( 'check passed' )
        except AssertionError as e:
            same = False
            print( e )
    f_cpp.close()
    f_py.close()
    if not same:
        print( '\n----------------- summary ------------------' )
    return same

def requireCloseEnough( item ):
    more = True
    num = 0
    while more:
        fn_cpp = f'/tmp/cpp_{item}{num}.txt'
        fn_py = f'/tmp/py_{item}{num}.txt'
        if os.path.exists( fn_cpp ) and os.path.exists( fn_py ):
            if checkCloseEnough( fn_cpp, fn_py ):
                print( f"Checking passed for {item}{num}" )
            else:
                print( f"Difference is detected for {item}{num}" )
            num += 1
        else:
            more = False

    fn_cpp = f'/tmp/cpp_{item}.txt'
    fn_py = f'/tmp/py_{item}.txt'
    if os.path.exists( fn_cpp ) and os.path.exists( fn_py ):
        if checkCloseEnough( fn_cpp, fn_py ):
            print( f"Checking passed for {item}" )
        else:
            print( f"Difference is detected for {item}" )

def checkEmbeddings():
    pass

def checkResult():
    sameFileContentList = ['same_as', 'well_defined_idx', 'samples', 'on', 'initial_state', 
            'masks', 'imasks', 'wav_lens', 'signals', 'count', 'clusters']
    closeEnoughList = ['segmentations', 'clean_segmentations', 'binarize_score', 
            'binarized_segmentations', 
            'trimmed', 'sum_trimmed', 'count_data', 'batch_waveform', 'embeddings',
            'norm_embeddings', 'dist', 'clusterRes', 'soft_clusters', 'hard_clusters',
            'clustered_segmentations', 'filtered_embeddings','aggregated_output',
            'aggregated_mask', 'overlapping_chunk_count', 'masks_in_aggregate',
            'scores_in_aggregate', 'to_diarization_activations',
            'cropped_activations', 'cropped_count','sorted_speakers', 'discrete_diarization']
    for item in checkList:
        print(f'\n======================================')
        print(f'======== CHECKING {item} =========')
        if item in sameFileContentList:
            requireSameFileContent( item )
        elif item in closeEnoughList:
            requireCloseEnough( item )

def main():
    if len( sys.argv ) > 1 and sys.argv[1] == 'clean':
        deleteResultFiles()
        return
    else:
        checkResult()


if __name__ == '__main__':
    main()
