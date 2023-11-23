def get_conv_frames(frames):
    conv = []
    frames_ = []
    for i in range(len(frames)):
        conv_ = []
        for j in range(len(frames)):
            conv_.append(frames[i]+frames[j])
            frames_.append(frames[i]+frames[j])
        conv.append(conv_)
    frames_ = sorted(list(set(frames_)))
    return conv, frames_

def get_gen_frames_indexed(frames):
    conv = []
    frames_ = []
    for i in range(len(frames)):
        conv_ = []
        for j in range(len(frames)):
            conv_.append(frames[i]+frames[j])
            frames_.append(frames[i]+frames[j])
        conv.append(conv_)
    frames_ = sorted(list(set(frames_)))
    for i in range (len(conv)):
        for j in range(len(conv[i])):
            conv[i][j] = frames_.index(conv[i][j])
    return conv, frames_

