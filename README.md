# AI Sudoku Solver using Computer Vision

This project uses OpenCV and a custom-trained Convolutional Neural Network (CNN) to automatically solve Sudoku puzzles from an image. The program identifies the Sudoku grid in a picture, recognizes the digits, solves the puzzle using a backtracking algorithm, and then annotates the original image with the solution.

## Demonstration

Here is an example of the solver in action:

| Unsolved Board | Solved Board |
| :---: | :---: |
| ![Unsolved Sudoku](https://github.com/savitra-roy/sudoku-vision-solver/blob/main/images/img6.jpg?raw=true) | ![Solved Sudoku](https://github.com/savitra-roy/sudoku-vision-solver/blob/main/solved/solved4.png?raw=true) |

## How It Works

The project follows a multi-stage pipeline to get from an input image to a solved puzzle.

### 1. Image Processing & Grid Detection
The first step is to find the Sudoku grid within the image and warp it into a flat, top-down view for easier analysis.

- **Preprocessing:** The image is converted to grayscale, a Gaussian blur is applied to reduce noise, and an adaptive threshold is used to create a clean binary image.
- **Contour Detection:** The largest four-sided contour in the image is identified, which is assumed to be the Sudoku grid.
- **Perspective Transform:** The four detected corners of the grid are used to apply a perspective transform, creating a perfectly square, straightened image of the puzzle.

### 2. A Novel Approach to Digit Isolation
A key challenge is distinguishing empty cells from cells containing digits, especially when grid lines are thick or there is image noise. Instead of relying on complex contour analysis for every cell, this project uses a more direct and efficient method:

- **Average Pixel Intensity:** Each cell is cropped to remove its borders, isolating the central area where a digit would be. The average pixel value (or intensity) of this central area is then calculated.
- **Thresholding:** If the average intensity is above a certain threshold, it indicates the presence of enough white pixels to be considered a digit. If it's below the threshold, the cell is marked as empty (0). This simple but effective technique is robust against minor noise and avoids the complexities of contour-based filtering for every cell.

### 3. Digit Recognition
Once a cell is identified as containing a number, a CNN is used to determine which digit it is.

- **Model:** A Convolutional Neural Network is trained on the classic MNIST dataset of handwritten digits.
- **Data Augmentation:** To improve the model's ability to recognize digits from real-world images (which are not as clean as the MNIST dataset), data augmentation techniques like random rotations, shifts, and zooms were applied during training. This makes the model more robust.
- **Prediction:** The isolated digit image from each cell is resized to 28x28 pixels and fed into the trained model for prediction.

### 4. Solving the Puzzle
With the digital 9x9 grid extracted, a classic backtracking algorithm is used to find the solution.

- **Backtracking:** This recursive algorithm works by trying to place a valid number (1-9) in an empty cell and then attempting to solve the rest of the puzzle. If it hits a dead end, it backtracks and tries the next number until a full solution is found.

### 5. Annotating the Solution
The final step is to display the solution on the original image.

- **Inverse Perspective Transform:** The inverse of the initial perspective transform matrix is calculated. This allows us to map the center coordinates of each solved cell on the flat grid back to their corresponding positions on the original, skewed image.
- **Drawing the Numbers:** The solved numbers are then drawn onto the original image at these calculated positions, with a font size that scales dynamically with the size of the detected grid.

## Shortcomings & Future Improvements

- **Image Quality:** The current pipeline is highly dependent on clear, well-lit images with a standard Sudoku grid. It may struggle with blurry images, heavy shadows, or unconventional grid designs.
- **Digit Recognition Accuracy:** While the augmented MNIST model performs well, it can still make mistakes, especially with printed fonts that differ significantly from the handwritten style of the training data. An invalid board due to a single misread number will cause the solver to fail.
- **Detection Method Brittleness:** The average pixel intensity method, while novel and efficient, can be sensitive to the `threshold` value and the amount of cropping (`margin`) applied to each cell. These values may need to be tuned for different types of images.

A significant future improvement would be to create a **custom dataset** by extracting and manually labeling digits from hundreds of different Sudoku puzzle images. Training the CNN on this highly specific data would dramatically improve recognition accuracy.

## Setup & Usage

To run this project, you will need Python and the following libraries:

- **OpenCV:** `pip install opencv-python`
- **NumPy:** `pip install numpy`
- **TensorFlow/Keras:** `pip install tensorflow`

1.  Clone the repository.
2.  Make sure you have the trained model file (`sudoku_digit_model.h5`) in the same directory.
3.  Run the main Python script, passing the filepath of your Sudoku image to the `pipe()` function.

```python
# Example usage
output_image = pipe('path/to/your/image.jpg')
cv2_imshow(output_image)
