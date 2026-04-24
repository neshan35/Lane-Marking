import cv2 # handles images
import os # lets python talk to folders
import numpy as np # allows you to understand large numeric matrix/ranges for the colors

input_folder = "images"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)





# goes through every image in input_folder (images)
for filename in os.listdir(input_folder):
    # safety check to make sure your opening an image file -- could add ".png", ".jpeg"
    if filename.lower().endswith((".jpg")):

        # builds actual paths to that file/image like images/img1.jpg
        path = os.path.join(input_folder, filename)
        # loads the image into memory
        img = cv2.imread(path)

        # if it can't read the image it will notify you, and skip it
        if img is None:
            print("Could not load:", filename)
            continue


        # ---------- HSV conversion -----------
        # switches photo from default RGB to HSV so we can extract white colors
        img = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ---------white detection -----------
        # white color range
        ## lower_white = np.array([0, 0, 140])
        ## upper_white = np.array([180, 90, 255])

        # just did a gray and white range instead
        mask1 = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 70, 255]))
        # 0, 0, 120 - 180, 40, 220
        mask2 = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 40, 255]))

        # mixed the two masks together into one
        mask = cv2.bitwise_or(mask1, mask2)


        # Before ROI (to see if its worth it)
        # cv2.namedWindow("HSV Mask (before ROI)", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("HSV Mask (before ROI)", 600, 400)
        # cv2.imshow("HSV Mask (before ROI)", mask)

        # -------------- ROI (region of interest) -------------------
        height, width = img.shape[:2]

        mask_roi = np.zeros_like(mask)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width, int(height * 0.55)),
            (0, int(height * 0.55))
        ]])

        cv2.fillPoly(mask_roi, polygon, 255)

        # apply ROI to mask
        mask = cv2.bitwise_and(mask, mask_roi)

        #----------- Fills In Gaps --------------------
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    

        # -------- save output --------
        # builds the path for that image into the output_folder so output
        out_path = os.path.join(output_folder, filename)
        # saves the image to output
        cv2.imwrite(out_path, mask)


        # -------- preview --------
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original", 600, 400)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask", 600, 400)
        # Lets you visually see results instantly.
        cv2.imshow("Original", img)
        cv2.imshow("Mask", mask)

        print("Press any key on image window to close...")
        # Stops progra
        # m until you press a key.
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # lets you know whats doen
        print("Processed:", filename)
