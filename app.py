import json
import uuid
from datetime import datetime
from threading import Thread
from flask import Flask, request#, url_for
from werkzeug.utils import secure_filename
#from flask_api import FlaskAPI, status, exceptions
from VideoEditor import VideoEditor

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4', 'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 # default no limit
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return "hello there !!\nthis server to process videos"

@app.route('/video/process/', methods=['GET', 'POST'])
def video_process():
    """
    process video
    """
    if request.method != 'POST':
        return "use 'POST' method"
        
    req = request.json
    if req is None or len(req) == 0:
        return "empty request. be sure that request sended as JSON"
    
    print('--- New Req ---')

    # read video input and mask and ouput
    input_video = req.get('input_video')
    mask_path = req.get('mask_path', None)
    output_video = req.get('output_video', None)

    # init
    test_video = VideoEditor(input_video, mask_path, output_video)
    
    # start process in new thread
    thr = Thread(target=test_video.hide_face) # args=[some_object])
    thr.start()

    # return output result
    res = dict()
    res['status'] = 'success'
    res['output_video'] = test_video.output_video
    
    return json.dumps(res, indent = 4)

@app.route('/video/upload/', methods=['GET', 'POST'])
def video_upload():
    """
    upload video
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file'] # .read() / .save('/tmp/foo')
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S___") + str(uuid.uuid4())
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('download_file', name=filename))
    return ''


if __name__ == "__main__":
    app.run(debug=True,threaded=True)
