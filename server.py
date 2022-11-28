import argparse
import os
import shutil
import tempfile
from http import HTTPStatus
from pathlib import Path

from flask import Flask, Response, request, render_template_string, send_file

from preprocessor import process_video as preprocess_video
from synthesizer import inference as sif
from video_inference import speech_synthesis

OUTPUT_PATH = '/shared/synthesised_video'

app = Flask(__name__)
args = None
synthesizer = None


@app.route('/')
def index():
    return render_template_string('''
        <html>
            <h2><u>Visual Speech Synthesis:</u></h2>
            
            <h3>Webcam</h3>
            <video id='video' autoplay></video>
            <br><br>
            <button id='start'>Record</button>
            <button id='stop'>Synthesise</button>
            
            <hr>
            
            <h3>Video Upload</h3>
            <form method="POST" action="/synthesise?download=1" enctype="multipart/form-data">
                <input type="file" name="video"/>
                <input type="submit" value="Synthesise"/>
            </form>
            
            <script>
                var video, startBtn, stopBtn, stream, recorder;
                video = document.getElementById('video');
                startBtn = document.getElementById('start');
                stopBtn = document.getElementById('stop');
                
                startBtn.onclick = startRecording;
                stopBtn.onclick = stopRecording;
                startBtn.disabled = true;
                stopBtn.disabled = true;
                
                navigator.mediaDevices.getUserMedia({
                    video: {width: 200, height: 500},
                    audio: true
                })
                .then(stm => {
                    stream = stm;
                    startBtn.removeAttribute('disabled');
                    video.srcObject = stream;
                }).catch(e => console.error(e));
                
                function startRecording() {
                    recorder = new MediaRecorder(stream, {
                        mimeType: 'video/webm'
                    });
                    recorder.start();
                    stopBtn.removeAttribute('disabled');
                    startBtn.disabled = true;
                }
                
                function stopRecording() {
                    recorder.ondataavailable = e => {
                        stopBtn.disabled = true;
                        stopBtn.textContent = "Please Wait..."
                    
                        var xhr = new XMLHttpRequest();
                        
                        xhr.addEventListener('load', function(event) {
                            if (xhr.status == 400) {
                                alert("Please record a clearer video");
                            } else {
                                window.open("/generated_video");
                            }
                            startBtn.removeAttribute('disabled');
                            stopBtn.textContent = "Synthesise"
                        });
                        
                        var formData = new FormData();
                        formData.append('video', e.data);
                        xhr.open('POST', '/synthesise');
                        xhr.send(formData);
                    };
                    recorder.stop();
                }
            </script>
        </html>
    ''')


@app.route('/generated_video')
def generated_video():
    return send_file(Path(OUTPUT_PATH).joinpath('generated_video.mp4'))


@app.route('/synthesise', methods=['POST'])
def synthesise():
    global args, synthesizer

    download = bool(request.args.get('download', 0))

    # upload file
    video_file = request.files['video']
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
    with open(temp_file.name, 'wb') as f:
        f.write(video_file.read())
    temp_file.seek(0)

    audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
    shutil.copyfile(args.audio_path, audio_file.name)

    output_directory = Path(OUTPUT_PATH)
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir()

    # preprocess video
    result = preprocess_video(
        process_index=0,
        video_path=temp_file.name,
        fps=args.fps,
        output_directory=output_directory,
        audio_preprocessing=args.use_audio_preprocessing,
        use_old_ffmpeg=args.use_old_ffmpeg,
        use_old_mouth_extractor=args.use_old_mouth_extractor,
        speaker_embedding_audio_file=audio_file
    )
    temp_file.close()
    audio_file.close()
    if result is None:
        return 'Failed to preprocess video', HTTPStatus.BAD_REQUEST

    # run synthesis
    speech_synthesis(
        synthesizer=synthesizer,
        video_directory=output_directory,
        combine_audio_and_video=True,
        save_alignments=True
    )

    if download:
        return send_file(output_directory.joinpath('generated_video.mp4'))

    return '', HTTPStatus.NO_CONTENT


@app.after_request
def add_header(r):
    """don't cache to reload the generated video"""
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    r.headers['Cache-Control'] = 'public, max-age=0'

    return r


def main():
    global args
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # set hparams from args
    sif.hparams.set_hparam('eval_ckpt', args.model_checkpoint)
    sif.hparams.set_hparam('img_height', args.image_height)
    sif.hparams.set_hparam('img_width', args.image_width)

    global synthesizer
    synthesizer = sif.Synthesizer(verbose=False)
    synthesizer.load()

    app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    """
    # requires webcam access which can only be done over https
    # use ngrok for https
    ngrok http 172.17.0.2:5000
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint')
    parser.add_argument('--image_height', type=int, default=50)
    parser.add_argument('--image_width', type=int, default=100)
    parser.add_argument('--audio_path')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--use_audio_preprocessing', action='store_true')
    parser.add_argument('--use_old_ffmpeg', action='store_true')
    parser.add_argument('--use_old_mouth_extractor', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()

    main()
