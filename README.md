# 🌌 HybridIndex 
HybridIndex is an easy to use Python library that lets you add a **retrieval engine** in your AI apps.

It uses and **hybrid search** (semantic + keyword) that outperforms both semantic search and keyword search individually.

With HybridIndex, you can build LLM-powered apps without the headaches of maintaining your own retrieval engine, while **outperforming** the current alternatives and being **free** to use.

<br>

# 📥 Installation

`pip install hybrid-index`

<br>

# 📙 Usage

To use the index, first create a new instance and provide it with a **name** and an **openai key** (to embed the documents).
```python
from hybrid_index import HybridIndex

index = HybridIndex('name', 'your-openai-key')
```


Then, we can start adding documents to the index. Each document in the list is a tuple containing an id (can be whatever you want, but keep it unique) and the document body:

```python
await index.add([
('a1', 'The sun rises in the east and sets in the west.'), 
('a2', 'Elephants are the largest land animals on Earth.'), 
('a3', 'Water boils at 100 degrees Celsius.'),
('a4', 'The capital of France is Paris.'),
('a5', 'The Great Wall of China is a famous landmark.')
])
```

Finally, we can **search** the index specifying the query and the number of results to retrieve (in our case 3):

```python
results = await index.query('What are the largest land animals on Earth?', 3)
```

The above code returns a list of the top 3 ids **ordered by similarity**:

```python
['a2', 'a4', 'a1']
```

The index can be serialized and deserialized (so that it can be saved) using pickle:
```python
pickled_index = index.serialize()

# After closing the session and saving the pickled index, we can load it back and rebuild the index

index2 = HybridIndex.deserialize(pickled_index, 'your-openai-key')
```

For security purposes your openai key is not pickled alongside your index, so it must be re-entered at deserialization time.

<br>

# ⚙️ How it works

This index uses a technique called **hybrid search** that aims to solve the shortcomings of semantic search (that is uncapable to find relevant keywords) and keyword search (that does not understand the meaning of the documents and query).

To achieve this goal, the HybridIndex stores two indices:
- A faiss index for semantic search with cosine similarity
- A modified **BM25O+** index that measure keyword similarity

The **BM25O+** index is a modified version of BM25+ that uses $\dfrac{corpus \ size}{frequency^2}$ to calculate the **IDF**. This formula has the effect of putting an enphasis on rare keywords while the semantic search takes care of the rest.

Both indexes are queried and each result is then combined using $\alpha \cdot cosine \ score + (1 - \alpha ) \cdot BM25O+ \ score$.

$\alpha$ is a parameter set by default at 0.7 (after a bit of experimentation), but can be chosen when initializing the index:

```python
index = HybridIndex('name', 'your-openai-key', a=0.6)
```

Finally, the top n results are returned to the user.

<br>

# 🪪 License

HybridIndex is licensed under the MIT License. See the LICENSE file for more details.
