from processing import get_combined_mask, display_mask, overlay_mask_on_image
import cv2 

def main():
    pass
    #step 1: segment background
    #step 2: remove cells to create noise background
    #step 3: train autoencoder x -> original image, y -> noise 
    
    image = cv2.imread("images/CFE001-0-M1-phase_img7150.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
    mask = get_combined_mask("masks/test.txt", image.shape)
    display_mask(mask)
    
    masked_image = overlay_mask_on_image(image, mask)
    display_mask(masked_image)
    
    subtracted_image = cv2.subtract(image, masked_image)
    display_mask(subtracted_image)


if __name__ == "__main__":
    main()