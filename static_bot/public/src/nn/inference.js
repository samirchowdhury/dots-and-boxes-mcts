import { encodeSnapshot } from "../game/encoding.js";

export function evaluate(model, snapshot) {
  const encoded = encodeSnapshot(snapshot);
  const { height, width } = encoded;
  const inputChannels = model.manifest.model.channels;
  const hiddenSize = model.manifest.model.hiddenSize;
  const residualBlocks = model.manifest.model.residualBlocks;

  let features = conv2dSame(
    encoded.tensor,
    height,
    width,
    inputChannels,
    weight(model, "stem_conv.weight"),
    shape(model, "stem_conv.weight"),
  );
  batchNormInPlace(model, "stem_bn", features, height, width, hiddenSize);
  reluInPlace(features);

  for (let block = 0; block < residualBlocks; block += 1) {
    const residual = features;
    let y = conv2dSame(
      features,
      height,
      width,
      hiddenSize,
      weight(model, `blocks.${block}.conv1.weight`),
      shape(model, `blocks.${block}.conv1.weight`),
    );
    batchNormInPlace(model, `blocks.${block}.bn1`, y, height, width, hiddenSize);
    reluInPlace(y);
    y = conv2dSame(
      y,
      height,
      width,
      hiddenSize,
      weight(model, `blocks.${block}.conv2.weight`),
      shape(model, `blocks.${block}.conv2.weight`),
    );
    batchNormInPlace(model, `blocks.${block}.bn2`, y, height, width, hiddenSize);
    for (let index = 0; index < y.length; index += 1) {
      y[index] = Math.max(0, y[index] + residual[index]);
    }
    features = y;
  }

  const policyFeatures = conv2dSame(
    features,
    height,
    width,
    hiddenSize,
    weight(model, "policy_conv.weight"),
    shape(model, "policy_conv.weight"),
  );
  batchNormInPlace(model, "policy_bn", policyFeatures, height, width, 2);
  reluInPlace(policyFeatures);
  const policyLogits = linear(
    policyFeatures,
    weight(model, "policy_linear.weight"),
    weight(model, "policy_linear.bias"),
    shape(model, "policy_linear.weight"),
  );
  const policy = maskedSoftmax(policyLogits, encoded.legalMask);

  const valueFeatures = conv2dSame(
    features,
    height,
    width,
    hiddenSize,
    weight(model, "value_conv.weight"),
    shape(model, "value_conv.weight"),
  );
  batchNormInPlace(model, "value_bn", valueFeatures, height, width, 1);
  reluInPlace(valueFeatures);
  const valueHidden = linear(
    valueFeatures,
    weight(model, "value_linear1.weight"),
    weight(model, "value_linear1.bias"),
    shape(model, "value_linear1.weight"),
  );
  reluInPlace(valueHidden);
  const [rawValue] = linear(
    valueHidden,
    weight(model, "value_linear2.weight"),
    weight(model, "value_linear2.bias"),
    shape(model, "value_linear2.weight"),
  );

  const priors = {};
  for (let index = 0; index < policy.length; index += 1) {
    if (encoded.legalMask[index] > 0) {
      priors[encoded.actionIds[index]] = policy[index];
    }
  }
  return {
    priors,
    value: Math.tanh(rawValue),
    policy,
    legalMask: encoded.legalMask,
  };
}

function weight(model, name) {
  const value = model.weights[name];
  if (!value) {
    throw new Error(`Missing model weight: ${name}`);
  }
  return value;
}

function shape(model, name) {
  return model.manifest.weights[name].shape;
}

function conv2dSame(input, height, width, inputChannels, kernel, kernelShape) {
  const [outputChannels, kernelHeight, kernelWidth, kernelInputChannels] = kernelShape;
  if (kernelInputChannels !== inputChannels) {
    throw new Error(
      `Conv input channel mismatch: kernel=${kernelInputChannels} tensor=${inputChannels}`,
    );
  }
  const output = new Float32Array(height * width * outputChannels);
  const rowPad = Math.floor(kernelHeight / 2);
  const colPad = Math.floor(kernelWidth / 2);

  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      for (let out = 0; out < outputChannels; out += 1) {
        let sum = 0;
        for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow += 1) {
          const inputRow = row + kernelRow - rowPad;
          if (inputRow < 0 || inputRow >= height) {
            continue;
          }
          for (let kernelCol = 0; kernelCol < kernelWidth; kernelCol += 1) {
            const inputCol = col + kernelCol - colPad;
            if (inputCol < 0 || inputCol >= width) {
              continue;
            }
            const inputBase = (inputRow * width + inputCol) * inputChannels;
            const kernelBase =
              ((out * kernelHeight + kernelRow) * kernelWidth + kernelCol) * inputChannels;
            for (let channel = 0; channel < inputChannels; channel += 1) {
              sum += input[inputBase + channel] * kernel[kernelBase + channel];
            }
          }
        }
        output[(row * width + col) * outputChannels + out] = sum;
      }
    }
  }
  return output;
}

function batchNormInPlace(model, prefix, tensor, height, width, channels) {
  const gamma = weight(model, `${prefix}.weight`);
  const beta = weight(model, `${prefix}.bias`);
  const mean = weight(model, `${prefix}.running_mean`);
  const variance = weight(model, `${prefix}.running_var`);
  const epsilon = model.manifest.model.batchNormEpsilon;

  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const base = (row * width + col) * channels;
      for (let channel = 0; channel < channels; channel += 1) {
        tensor[base + channel] =
          ((tensor[base + channel] - mean[channel]) / Math.sqrt(variance[channel] + epsilon)) *
            gamma[channel] +
          beta[channel];
      }
    }
  }
}

function linear(input, weights, bias, weightShape) {
  const [outputSize, inputSize] = weightShape;
  if (input.length !== inputSize) {
    throw new Error(`Linear input mismatch: weight=${inputSize} tensor=${input.length}`);
  }
  const output = new Float32Array(outputSize);
  for (let out = 0; out < outputSize; out += 1) {
    let sum = bias[out];
    const weightBase = out * inputSize;
    for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
      sum += input[inputIndex] * weights[weightBase + inputIndex];
    }
    output[out] = sum;
  }
  return output;
}

function maskedSoftmax(logits, legalMask) {
  let maxLogit = -Infinity;
  for (let index = 0; index < logits.length; index += 1) {
    if (legalMask[index] > 0 && logits[index] > maxLogit) {
      maxLogit = logits[index];
    }
  }
  const probabilities = new Float32Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (legalMask[index] > 0) {
      const value = Math.exp(logits[index] - maxLogit);
      probabilities[index] = value;
      total += value;
    }
  }
  const normalizer = Math.max(total, 1.0e-12);
  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] /= normalizer;
  }
  return probabilities;
}

function reluInPlace(values) {
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] < 0) {
      values[index] = 0;
    }
  }
}
