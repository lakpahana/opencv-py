import cv2
import matplotlib.pyplot as plt

# Load the pre-trained model
net = cv2.dnn.readNet("path_to_pretrained_model.weights", "path_to_model_configuration.cfg")

# Load the classes
with open("path_to_classes.txt", "r") as f:
    classes = f.read().strip().split("\n")

# Load the image
image = cv2.imread("fruit.jpg")

# Prepare the image for the model
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set input for the network
net.setInput(blob)

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Forward pass
outputs = net.forward(output_layers)

# Process the detections
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # Get the coordinates of the detected object
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            # Draw the bounding box around the object
            cv2.rectangle(image, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the image with detected objects
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
