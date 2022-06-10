from PIL import Image, ImageDraw, ImageSequence, ImageFont, ImageFilter
import io

im = Image.open('gifs/old_version_gifs/500_-60.gif')
game_msg = "500 trained games"
score_msg = "total score: -60"
new_location = 'gifs/old_version_gifs/game_1.gif'


font_size = 40
font = font_size#ImageFont.truetype("calibri.ttf", font_size)
color = (160, 160, 160)
dimensions = (420, 280)
offset_y = 20

offset_x = 50


# A list of the frames to be outputted
frames = []
# Loop over each frame in the animated image
for frame in ImageSequence.Iterator(im):
    frame = frame.resize(dimensions, Image.ANTIALIAS)

    draw = ImageDraw.Draw(frame)

    game_location = tuple([(dimensions[0]-draw.textsize(game_msg)[0])/2, (dimensions[1]-draw.textsize(game_msg)[1])/2 - offset_x])
    score_location = tuple([(dimensions[0]-draw.textsize(score_msg)[0])/2, (dimensions[1]-draw.textsize(score_msg)[1])/2 - offset_x + offset_y])

    # Draw the text on the frame
    draw.text(game_location, game_msg, size=font, align="center", fill=color)
    draw.text(score_location, score_msg, size=font, align="center", fill=color)
    del draw

    frame = frame.convert("RGB")
    frame = frame.filter(ImageFilter.SHARPEN)
    # However, 'frame' is still the animated image with many frames
    # It has simply been seeked to a later frame
    # For our list of frames, we only want the current frame

    # Saving the image without 'save_all' will turn it into a single frame image, and we can then re-open it
    # To be efficient, we will save it to a stream, rather than to file
    b = io.BytesIO()
    frame.save(b, format="GIF")
    frame = Image.open(b)

    # Then append the single frame image to a list of frames
    frames.append(frame)
# Save the frames as a new image
frames[0].save(new_location, save_all=True, append_images=frames[1:])