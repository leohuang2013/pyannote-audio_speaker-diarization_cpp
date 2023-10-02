from pyannote.audio import Pipeline
from hf_token import *
import datetime
import time
import sys
import wave
import contextlib

import rich


def diarize( wavFile, min_speakers = None, max_speakers = None ):

    try:
        with contextlib.closing(wave.open(wavFile,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print( f'Diarize wav file: {wavFile} with duration: {duration}' )
    except Exception as e:
        print( f'Failed to open wav file: {e}' )
        return None
    
    # get the start datetime
    st = datetime.datetime.now()

    print( "load model" )
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=hf_auth_token)
    #pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
    #                                    cache_dir="/Users/leo/.cache/torch/pyannote")

    print( "apply pipeline" )
    # apply pretrained pipeline
    if min_speakers == None or max_speakers == None or \
         min_speakers == 0 or max_speakers == 0:
        '''
        If do not know the number of speaker 
        '''
        diarization = pipeline( wavFile )
    else:
        print( f"speakers info: min={min_speakers}, max={max_speakers}" )
        if min_speakers == max_speakers or min_speakers > max_speakers:
            '''
            If know exactly the number of speaker
            '''
            diarization = pipeline("audio.wav", num_speakers=min_speakers)
        elif min_speakers > max_speakers: # wrong config
            diarization = pipeline( wavFile )
        else:
            '''
            If know min and max of the number of speaker 
            '''
            diarization = pipeline("audio.wav", min_speakers=min_speakers, max_speakers=max_speakers)

    print('----------------------------------------------------')
    result = []
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        result.append({ "start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": f"{speaker}" })
    print('----------------------------------------------------')

    # get the end datetime
    et = datetime.datetime.now()

    # get execution time
    elapsed_time = et - st
    print('Time cost:', elapsed_time, 'seconds')

    return result

def main():
    if len( sys.argv ) < 2 :
        print( 'Wav file needed' )
        return

    wavFile = sys.argv[1]
    diarize( wavFile )

if __name__ == '__main__':
    main()
