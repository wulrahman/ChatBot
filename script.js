const MAX_SEQUENCE_LENGTH = 20;
// const VOCAB_SIZE = 17;
const EMBEDDING_DIM = 256;
const LSTM_UNITS = 256;

let model;
let wordIndex = {};

function calculateVocabularySize(trainingData) {
    const uniqueWords = new Set();

    // Iterate through training data and extract unique words
    trainingData.forEach(line => {
        const [question, answer] = line.split('|');
        const words = question.split(' ').concat(answer.split(' '));
        words.forEach(word => {
            uniqueWords.add(word);
        });
    });

    // Return the size of unique words set
    return uniqueWords.size;
}


const training_data = [
    "What is your name?|My name is John.",
    "How old are you?|I am 25 years old.",
    "Where are you from?|I am from New York.",
     "How are you?|I'm doing well, thank you.",
    "What do you like to do for fun?|I enjoy reading and hiking.",
    "Can you tell me about your family?|Sure, I have two siblings and loving parents.",
    "What is your favorite color?|My favorite color is blue.",
    "Do you have any pets?|Yes, I have a dog named Max.",
    "What is your favorite food?|I love pizza!",
    "What is your dream job?|I've always wanted to be a pilot.",
    "What is your favorite movie?|My favorite movie is The Shawshank Redemption.",
    "How do you spend your weekends?|I usually go hiking or spend time with friends.",
    "Do you like traveling?|Yes, I love exploring new places.",
    "What languages do you speak?|I speak English and Spanish.",
    "What is your favorite book?|My favorite book is To Kill a Mockingbird.",
    "What are your hobbies?|I enjoy painting and playing guitar.",
    "Do you prefer coffee or tea?|I prefer tea.",
    "What is your zodiac sign?|I'm a Leo.",
    "What is your favorite season?|I love autumn.",
    "What is your favorite holiday?|Christmas is my favorite holiday.",
    "What is your favorite sport?|I enjoy playing basketball.",
    "What is your favorite dessert?|I love chocolate cake.",
    "What is your favorite music genre?|I enjoy listening to rock music.",
    "What is your favorite animal?|I love dolphins.",
    "What is your favorite place to relax?|I enjoy relaxing at the beach.",
    "Do you have any siblings?|Yes, I have one older sister.",
    "What is your favorite hobby?|My favorite hobby is gardening.",
    "What is your favorite subject in school?|I enjoy studying mathematics.",
    "What is your favorite TV show?|I love watching Friends.",
    "What is your favorite color?|My favorite color is green.",
    "What is your favorite movie genre?|I enjoy watching comedy movies.",
    "Do you have any pets?|Yes, I have a cat named Luna.",
    "What is your favorite place to visit?|I love visiting museums.",
    "What is your favorite sport to watch?|I enjoy watching soccer.",
    "What is your favorite holiday destination?|I love going to the mountains.",
    "What is your favorite season of the year?|My favorite season is spring.",
    "What is your favorite type of cuisine?|I love Italian food.",
    "Do you enjoy cooking?|Yes, I love experimenting with new recipes.",
    "What is your favorite outdoor activity?|I enjoy going camping.",
    "What is your favorite indoor activity?|I like reading books.",
    "Do you prefer summer or winter?|I prefer winter.",
    "What is your favorite type of music?|I enjoy listening to classical music.",
    "Do you have any allergies?|Yes, I'm allergic to peanuts.",
    "What is your favorite beverage?|My favorite beverage is iced coffee.",
    "What is your favorite childhood memory?|My favorite childhood memory is going to the beach with my family.",
    "What is your favorite restaurant?|I love dining at a local sushi restaurant.",
]

const VOCAB_SIZE = calculateVocabularySize(training_data);

preprocessTrainingData(training_data);


async function trainModel() {

    const inputSequences = [];
    const outputSequences = [];

    training_data.forEach(line => {
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

    await model.fit(inputTensor, outputTensor, {
        epochs: 1000,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.getElementById('training-log').innerText += `Epoch ${epoch + 1}: Loss - ${logs.loss.toFixed(4)}, Accuracy - ${logs.acc.toFixed(4)}\n`;
            }
        }
    }).then(async () => {

        model.save();
        const sampleInput = "How are you!";

        // Load the model
        // Generate response for the sample input
        const response = await generateResponse(sampleInput);
        console.log('Sample Input:', sampleInput);
        console.log('Response:', response);

        // const modelData = await model.save('downloads://my-model');
        const saveResult = await model.save('localstorage://my-model');

        document.getElementById('training-status').innerText = 'Training status: Completed';

    });
   
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

const trainingData = document.getElementById('training-data');
trainingData.addEventListener('input', async (event) => {
    const value = trainingData.value; // Get the value of the input element
    const response = await generateResponse(value);
    console.log('Sample Input:', value);
    console.log('Response:', response);
});
// Sample input
const sampleInput = "How are you!";

const downloadBtn = document.getElementById('downloadBtn').addEventListener('click', async () => {
    const modelData = await model.save('downloads://my-model');
});


const trainModelBtn = document.getElementById('trainModelBtn').addEventListener('click', async () => {
    trainModel();
});

const createModelBtn = document.getElementById('createModelBtn').addEventListener('click', async () => {
    model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: VOCAB_SIZE, outputDim: EMBEDDING_DIM, inputLength: MAX_SEQUENCE_LENGTH }));
    model.add(tf.layers.lstm({ units: LSTM_UNITS, returnSequences: true }));
    model.add(tf.layers.dense({ units: VOCAB_SIZE, activation: 'softmax' }));

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: 'adam',
        metrics: ['accuracy']
    });
});

const loadModelBtn = document.getElementById('loadModelBtn').addEventListener('click', async () => {
    model = await tf.loadLayersModel('localstorage://my-model');
    const response = await generateResponse(sampleInput);
    // model.save('my_model.keras');
    console.log('Sample Input:', sampleInput);
    console.log('Response:', response);
});

const loadModelFileBtn = document.getElementById('loadModelFileBtn').addEventListener('click', async () => {
    model = await tf.loadLayersModel('./my-model.json');
    const response = await generateResponse(sampleInput);
    // model.save('my_model.keras');
    console.log('Sample Input:', sampleInput);
    console.log('Response:', response);
});