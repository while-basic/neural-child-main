Connect the AI's learned vocabulary from obsidian to @main.py. 
Give the AI the perception of time. It should know what time it is based off the computers clock, know how long its been since I last talked to it (AI should be aware of how long its been since the last chat and express emotions about it).



Add new neural networks to @main.py. Sadness, Anger, Disgust, Anticipation, Love (Joy + Trust), Guilt (Trust + Anticipation), Hope (Trust + Anticipation), Regret (Sadness + Disgust).

Create a chat interface for interacting with the AI. When interacting with the AI, it should consider the entire chat conversation (stored with chromadb) before calculating all the emotions combined (this should be processed by the neural network) and responding to the user.

The AI should also be learning words and adding them to the Vocabulary. These words should be added because they are not found in the AI's vocabulary (also stored in chromadb).

From the words, it should generate dreams, influenced by the new words it learned from chat conversations (found in )