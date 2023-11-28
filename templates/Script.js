const imageInput = document.getElementById("imageInput");
const classifyButton = document.getElementById("classifyButton");
const selectedImage = document.getElementById("selectedImage");
const predictionElement = document.getElementById("prediction");

imageInput.addEventListener("change", function () {
  const file = imageInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      selectedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});

classifyButton.addEventListener("click", function () {
  const file = imageInput.files[0];
  if (!file) {
    alert("Please select an image.");
    return;
  }

  // Prepare the image for uploading to the server (you may need to adjust this part)
  const formData = new FormData();
  formData.append("image", file);

  // Send the image to your server for processing using fetch or XMLHttpRequest
  fetch("/classify", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      predictionElement.textContent = `Prediction: ${data.prediction}`;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
});
