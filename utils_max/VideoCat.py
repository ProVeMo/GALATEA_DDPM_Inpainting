import natsort
import imageio
import os
from PIL import Image
from moviepy.editor import VideoFileClip, clips_array


def video_cat(vid1_path: str, vid2_path: str, output_path: str, video_name: str = 'cat_video'):
    if vid1_path and vid2_path and output_path:
        clip1 = VideoFileClip(vid1_path)  # .margin(10)  # add 10px contour
        clip2 = VideoFileClip(vid2_path)
        final_clip = clips_array([[clip1, clip2]])
        final_clip.write_videofile(os.path.join(output_path, f"{video_name}.mp4"))
    else:
        assert False, 'VideoPath muss angegeben werden'


def video_cat_4(vid1_path: str, vid2_path: str, vid3_path: str, vid4_path: str, output_path: str,
                video_name: str = 'cat_video'):
    clip1 = VideoFileClip(vid1_path)
    clip2 = VideoFileClip(vid2_path)
    clip3 = VideoFileClip(vid3_path)
    clip4 = VideoFileClip(vid4_path)
    final_clip = clips_array([[clip1, clip2], [clip3, clip4]])
    final_clip.write_videofile(os.path.join(output_path, f"{video_name}.mp4"))


def directory_to_video(input_dir_path: str, output_path: str, video_name: str = 'vid'):
    if input_dir_path and output_path:
        images = []
        for i in natsort.natsorted(os.listdir(input_dir_path)):
            if i.endswith(".png"):
                images.append(imageio.imread(os.path.join(input_dir_path, i)))
        images.reverse()
        video_file_path = os.path.join(output_path, f"{video_name}.mp4")
        writer = imageio.get_writer(video_file_path, fps=25)
        for image in images:
            writer.append_data(image)
        writer.close()
    else:
        assert False, 'Kein Korrekter Pfad angegeben'


def prepare_images():
    from TempCycle.data.sampled_seq_DL import ImageTransform
    from utils.TensorImageConverter import batch_to_image
    transform = ImageTransform(256)

    path = 'F:/Masterarbeit/Datasets/Sandbox_results/test/test3'
    out_path = 'F:/Masterarbeit/Datasets/Sandbox_results/transformed/Park/test2'
    for i in natsort.natsorted(os.listdir(path)):
        image = Image.open(os.path.join(path, i))
        image_tensor = transform.transform(image)
        transformed_image = batch_to_image(image_tensor, True)[0]
        transformed_image.save(os.path.join(out_path, f'{i}.png'))


if __name__ == '__main__':
    # prepare_images()
    # directory_to_video('F:/Masterarbeit/Datasets/Sandbox_results/transformed/Ipmi/test1', 'F:/Masterarbeit/Datasets/Sandbox_results/transformed/Ipmi', 'reverse_test1')
    """vid1_path = 'F:/Masterarbeit/Datasets/tempDataset/results/Park/run_79/Run79_E79_B0.mp4'
    vid2_path = 'F:/Masterarbeit/Datasets/Sandbox_results/transformed/Park/reverse_test2.mp4'
    output_path = 'F:/Masterarbeit/Datasets/tempDataset/results/Park/run_79/cated'
    video_name = 'Run79_E79_B0.mp4'
    video_cat(vid1_path, vid2_path, output_path, video_name)"""

    vid1_path = 'F:/Masterarbeit/Datasets/tempDataset/results/Park/run_79/Run79_E59_B0.mp4'
    vid2_path = 'F:/Masterarbeit/Datasets/tempDataset/results/Park/run_79/Run79_E69_B0.mp4'
    vid3_path = 'F:/Masterarbeit/Datasets/tempDataset/results/Park/run_79/Run79_E75_B0.mp4'
    vid4_path = 'F:/Masterarbeit/Datasets/tempDataset/results/Park/run_79/Run79_E79_B0.mp4'
    output_path = 'F:/Masterarbeit/Datasets/tempDataset/results/Park/run_79/cated'
    video_name = 'cat59_79.mp4'
    video_cat_4(vid1_path, vid2_path, vid3_path, vid4_path, output_path, video_name)
