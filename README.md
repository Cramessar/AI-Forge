# 🧠 AI Forge

**AI Forge** is a locally-powered desktop AI app I built to make using models like `mistral`, `llama2`, or `codellama` easier, faster, and more fun—without relying on API keys or an internet connection. If you've ever thought "I just want to run a model without jumping through five cloud hoops," welcome. You’re among friends now.

This app uses [Ollama](https://ollama.com) under the hood and is fully local-first. It’s meant to be simple for general users, powerful for devs, and welcoming for anyone who wants to explore AI—without a monthly bill.

---

## 🚀 Why I Built This

AI is powerful. But for a lot of people, it’s also confusing, expensive, or locked behind services they don’t control. I wanted to change that.

So I built **AI Forge** to:
- Make **local AI** more accessible for everyone
- Let folks generate, experiment, and create **without needing the cloud**
- Support amazing projects like **Ollama**, which make running models easy
- Build something the community can shape and grow together

This is a project I plan to keep working on. But honestly? The best version of AI Forge will come from more than just me. Your perspective, ideas, and skills can take this to places I haven’t imagined.

Together, we can make AI tools that **belong to the people**, not the platforms. Let's bring AI to the masses—no gatekeepers, no paywalls, no problem.

---

## 🧩 Features

- 🧠 Model switching via Ollama (`mistral`, `codellama`, etc.)
- 🔄 Real-time streaming output (with style)
- 💡 Syntax-highlighted code blocks (yes, finally readable code!)
- 🛠️ CPU / GPU backend benchmarking + tuning
- 📥 PDF, Markdown, TXT file importing
- 💾 Automatic local history caching (via SQLite)
- 📜 Manual session save/load
- 🎨 Dark-mode friendly with fully themed UI

Basically, it’s like ChatGPT met a local dev tool, drank some Red Bull, and decided to move off-grid.

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/cramessar/ai-forge.git
cd ai-forge

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is installed
# Then download a model like mistral
ollama run mistral

# Launch the app
python app.py
```

Note: Runs best on Python 3.10+ (and yes, I tested this more times than I’ve tested my own patience).

---

## 👤 Who This Is For

### 🎓 General Users
- Students, bloggers, storytellers, curious humans
- Want to write, summarize, or brainstorm without the tech overhead

### 🧑‍💻 Advanced Users
- Developers, researchers, tinkerers
- Want full control over models, backends, and configurations

---

## 🛠 How to Use It

1. **Pick a Model**
   - Use the 🧠 dropdown
   - I recommend `mistral` for general tasks or `codellama` for dev work

2. **Type Your Prompt**
   - Example:  
     - "Write a pitch for a cyberpunk bagel startup"  
     - "Explain quantum entanglement like I’m five"

3. **Click Generate**
   - Sit back and watch the AI do its thing

4. **Optional Tweaks**
   - Temperature = creativity
   - Max Tokens = response length

5. **Import Files**
   - PDFs, markdown, and text files are fair game

6. **Save or Load**
   - Save a session as `.json`  
   - Reopen later for ongoing projects

7. **Forget to Save?**
   - Don’t worry, I added SQLite caching so your stuff sticks around  
   - (Unless you delete the file. Then… well, don’t do that.)

---

## 🧪 Performance Tweaks

- Open 🛠️ **Settings**
- Choose **CPU**, **GPU**, or **Auto**
- Run **Benchmark** to test which is faster
- GPU's faster? Great. CPU’s faster? Still great.  
- Auto just picks the winner for you, like a non-judgmental referee.

---

## 🗃 Local History (SQLite FTW)

All prompts + responses are stored automatically in `history.db`:

- Appears in the sidebar
- Saved silently in the background
- Doesn’t require saving a session manually

Think of it as a memory that doesn’t forget—even when you do.

---

## ✍️ Developer Notes

- All config lives in `config.json`
- Backend tuning and session persistence is built-in
- Database logic lives in `db.py` (SQLite by default, swappable)
- Models are served using **Ollama**—so get cozy with `ollama run` and `ollama list`

Pro tip: yes, you can theme it, mod it, or plug in your own models.

---

## 🌍 Community Call

AI Forge is open source, and I’d love your help.

- Got ideas for features?
- Want to add plugin support?
- Got a custom use case?

Whether you're writing code, spotting bugs, designing UI, or suggesting ideas—you’re welcome here.

Let’s build something that helps people **create freely, learn deeply, and explore widely**—without being held hostage by cloud pricing.

---

## ❤️ Thanks & Acknowledgments

- [Ollama](https://ollama.com) — Seriously, go support them. They’re doing incredible work.
- Pygments — For making code look less like chicken scratch.
- Markdown2 — For doing what Markdown does best: just working.

---

## 📚 Roadmap / Coming Soon

- Custom model profiles
- Queryable local history
- Plugin system (Notion? VS Code? Terminal dragons?)
- Community prompt sharing
- Batch generation

---

### 🧙 Final Thoughts

> “Not all who wander are lost... but some are just looking for a local AI tool that respects their GPU.”

Thanks for checking out AI Forge.  
Now go forth and generate something epic.

— _Chris_
