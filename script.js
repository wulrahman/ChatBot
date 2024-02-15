const MAX_SEQUENCE_LENGTH = 20;
const VOCAB_SIZE = 10000;
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
        epochs: 50,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.getElementById('training-log').innerText += `Epoch ${epoch + 1}: Loss - ${logs.loss.toFixed(4)}, Accuracy - ${logs.acc.toFixed(4)}\n`;
            }
        }
    });

    await model.save('downloads://chatbot_model');

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

