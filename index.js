const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const Upscaler = require("upscaler/node");

const fileTypes = { ffd8ffe0: "jpeg", 89504e47: "png" };

const checkImageType = (image) => {
  const imageToString = image.toString("hex", 0, 4);
  return fileTypes[imageToString];
};

const enhancedString = (path) => {
  const pathName = path.split(".").join("");
  return pathName + "-" + "enhanced";
};

const main = async () => {
  console.log(process.argv[2]);
  const upscaler = new Upscaler();
  const image = tf.node.decodeImage(fs.readFileSync(process.argv[2]), 3);
  const fileType = checkImageType(fs.readFileSync(process.argv[2]));

  const tensor = await upscaler.upscale(image);
  let upscaledTensor;

  if (fileType === "png") {
    upscaledTensor = await tf.node.encodePng(tensor);
  } else {
    upscaledTensor = await tf.node.encodeJpeg(tensor);
  }

  const enhancedFilePath = `${enhancedString(process.argv[2])}.${fileType}`;

  fs.writeFileSync(enhancedFilePath, upscaledTensor);
  console.log("wrote the file image as ", fileType);

  // dispose the tensors!
  image.dispose();
  tensor.dispose();
  process.exit();
};

main();
