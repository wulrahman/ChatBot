const MAX_SEQUENCE_LENGTH = 20;
const VOCAB_SIZE = 17;
const EMBEDDING_DIM = 256;
const LSTM_UNITS = 256;

let model;
let wordIndex = {};

async function trainModel() {
    const trainingData = document.getElementById('training-data').value.split('\n').map(line => line.trim()).filter(line => line !== '');

    preprocessTrainingData(trainingData);

    const inputSequences = [];
    const outputSequences = [];

    trainingData.forEach(line => {
        const [question, answer] = line.split('|');
        const inputSeq = tokenizeSentence(question);
        const outputSeq = tokenizeSentence(answer);
        inputSequences.push(inputSeq);
        outputSequences.push(outputSeq);
    });

    const paddedInputSequences = padSequences(inputSequences);
    const paddedOutputSequences = padSequences(outputSequences);

    const inputTensor = tf.tensor2d(paddedInputSequences);
    const outputTensor = tf.tensor3d(oneHotEncode(paddedOutputSequences, VOCAB_SIZE), [paddedOutputSequences.length, MAX_SEQUENCE_LENGTH, VOCAB_SIZE]);

    model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: VOCAB_SIZE, outputDim: EMBEDDING_DIM, inputLength: MAX_SEQUENCE_LENGTH }));
    model.add(tf.layers.lstm({ units: LSTM_UNITS, returnSequences: true }));
    model.add(tf.layers.dense({ units: VOCAB_SIZE, activation: 'softmax' }));

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: 'adam',
        metrics: ['accuracy']
    });

    await model.fit(inputTensor, outputTensor, {
        epochs: 200,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.getElementById('training-log').innerText += `Epoch ${epoch + 1}: Loss - ${logs.loss.toFixed(4)}, Accuracy - ${logs.acc.toFixed(4)}\n`;
            }
        }
    });

    
    // Sample input
    const sampleInput = "How are you!";

    // Load the model
    // Generate response for the sample input
    const response = await generateResponse(sampleInput);
    console.log('Sample Input:', sampleInput);
    console.log('Response:', response);


    // await model.save('downloads://chatbot_model');

    document.getElementById('training-status').innerText = 'Training status: Completed';
}

function oneHotEncode(sequences, vocabSize) {
    const oneHotSequences = [];
    sequences.forEach(seq => {
        const oneHotSeq = tf.oneHot(seq, vocabSize).arraySync();
        oneHotSequences.push(oneHotSeq);
    });
    return oneHotSequences;
}

function padSequences(sequences) {
    return sequences.map(seq => {
        if (seq.length > MAX_SEQUENCE_LENGTH) {
            seq.splice(0, seq.length - MAX_SEQUENCE_LENGTH);
        }
        if (seq.length < MAX_SEQUENCE_LENGTH) {
            seq = new Array(MAX_SEQUENCE_LENGTH - seq.length).fill(0).concat(seq);
        }
        return seq;
    });
}

function padSequence(sequence) {
    if (sequence.length > MAX_SEQUENCE_LENGTH) {
        sequence.splice(0, sequence.length - MAX_SEQUENCE_LENGTH);
    }
    if (sequence.length < MAX_SEQUENCE_LENGTH) {
        sequence = new Array(MAX_SEQUENCE_LENGTH - sequence.length).fill(0).concat(sequence);
    }
    return sequence;
}
function preprocessTrainingData(data) {
    let index = 1;
    data.forEach(line => {
        const [question, answer] = line.split('|');
        const words = question.split(' ').concat(answer.split(' '));
        words.forEach(word => {
            if (!wordIndex[word]) {
                wordIndex[word] = index;
                index++;
            }
        });
    });
}

function tokenizeSentence(sentence) {
    const tokens = sentence.split(' ');
    const sequence = [];
    tokens.forEach(token => {
        sequence.push(wordIndex[token]);
    });
    return sequence;
}


// // Function to load the saved model
// async function loadModel() {
//     model = await tf.loadLayersModel('downloads://chatbot_model');
//     console.log('Model loaded');
// }

// Function to generate a response for the input
async function generateResponse(input) {
    // Tokenize the input
    const inputSeq = tokenizeSentence(input);
    
    // Pad the input sequence
    const paddedInputSeq = padSequence(inputSeq);
    
    // Convert the padded sequence to a tensor
    const inputTensor = tf.tensor2d(paddedInputSeq, [1, MAX_SEQUENCE_LENGTH]); // Shape should be [1, sequence_length]
    
    // Perform inference using the model
    const outputTensor = model.predict(inputTensor);
    
    // Decode the output sequence to get the response
    const responseIndex = outputTensor.argMax(2).dataSync();    
    const response = reverseTokenizeSentence(responseIndex);
    
    return response;
}


// Function to reverse tokenize the output sequence
function reverseTokenizeSentence(sequence) {
    
    const words = [];
    sequence.forEach(index => {
        for (const [word, idx] of Object.entries(wordIndex)) {
            if (idx === index) {
                words.push(word);
                break;
            }
        }
    });
    
    console.log('Decoded Words:', words);
    
    return words.join(' ');
}


// Sample input
// const sampleInput = "Hello!";

// // Load the model
// loadModel().then(async () => {
//     // Generate response for the sample input
//     const response = await generateResponse(sampleInput);
//     console.log('Sample Input:', sampleInput);
//     console.log('Response:', response);
// });
