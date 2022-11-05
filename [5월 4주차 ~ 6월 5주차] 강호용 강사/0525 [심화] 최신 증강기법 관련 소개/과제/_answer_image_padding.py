from PIL import Image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # paste 이미지 붙이기(추가할 이미지 , 붙일 위치(가로, 세로))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        # 새로운 이미지 생성
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


im = Image.open("./or_image.jpeg")
im_new = expand2square(im, (0, 0, 0)).resize((224, 224))
im_new.save('./astronaut_expand_square.png', quality=100)
