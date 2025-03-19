import image_processing as imp
import clip_test as ct

def main():
    dfdim = (224,224)
    image1 = imp.read_image(
        "res/test_image_house.jpg", 
        flag=imp.IOFlags.GRAYSCALE,
        dim=dfdim
    ) 
    image2 = imp.read_image(
        "res/test_image_castle.jpg", 
        flag=imp.IOFlags.GRAYSCALE,
        dim=dfdim
    )


    imfiltered1 = imp.lowpass_filter(image1)
    imfiltered2 = imp.xdog(image1, sigma=0.5, k=100, gamma=.7, epsilon=0.3, phi=1)
    imfiltered2 = imp.otsu_thresholding(imfiltered2)

    imp.display_image(imfiltered1)
    imp.display_image(imfiltered2)
    print(ct.compare_images(image1, imfiltered1), imp.ssim(image1, imfiltered1))
    print(ct.compare_images(image1, imfiltered2), imp.ssim(image1, imfiltered2))
    print(ct.compare_image_text(
        image1, 
        "A two-story suburban house with a light gray exterior, dark brown roof, and a wooden front door."
        )
    )
    print(ct.compare_image_text(
        image1, 
        "A two-story house." 
        )
    )

    #imp.save_image("out/lowfreq_house.jpg", imfiltered1)
    #imp.save_image("out/xdogthresh_house.jpg", imfiltered2)

if __name__ == "__main__":
    main()