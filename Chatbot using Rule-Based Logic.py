import re
import datetime

rules = {
    r'\b(hi|hello|hey|howdy)\b'                        : "Hey there! How can I help you?",
    r'\b(bye|goodbye|exit|quit|see you)\b'              : "Bye! Take care :)",
    r'\b(how are you|how do you do|hows it going)\b'   : "I'm doing good, thanks for asking! I'm just a bot though lol",
    r'\b(your name|who are you|what are you)\b'         : "I'm a simple chatbot made using Python rules, nothing fancy.",
    r'\b(what can you do|help|commands|options)\b'      : (
        "Here's what I understand:\n"
        "  - greetings (hi, hello, hey)\n"
        "  - current time and date\n"
        "  - tell a joke\n"
        "  - simple math like 5 + 3\n"
        "  - some basic FAQs\n"
        "Just type normally and I'll try my best!"
    ),
    r'\b(time|what time|current time)\b'                : "__TIME__",
    r'\b(date|today|what day|whats the date)\b'         : "__DATE__",
    r'\b(joke|funny|make me laugh|tell me a joke)\b'   : (
        "Why don't scientists trust atoms?\n"
        "Because they make up everything 😂"
    ),
    r'\b(thanks|thank you|thx|ty)\b'                   : "No problem!",
    r'\b(weather)\b'                                    : "I can't check live weather, but try weather.com or just Google it.",
    r'\b(how old|your age|age)\b'                      : "I was born the moment this script ran lol",
    r'\b(who made you|who built you|creator)\b'        : "A student made me as part of an AI internship project!",
    r'(\d+\s*[\+\-\*\/]\s*\d+)'                        : "__MATH__",
}


def get_reply(msg):
    msg = msg.lower().strip()

    if not msg:
        return "Say something!"

    for pattern, reply in rules.items():
        if re.search(pattern, msg):

            if reply == "__TIME__":
                t = datetime.datetime.now().strftime("%I:%M %p")
                return f"Current time is {t}"

            if reply == "__DATE__":
                d = datetime.datetime.now().strftime("%A, %B %d, %Y")
                return f"Today is {d}"

            if reply == "__MATH__":
                try:
                    expr = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', msg).group(1)
                    ans = eval(expr)
                    return f"{expr.strip()} = {ans}"
                except:
                    return "Couldn't calculate that, try something like '4 + 6'"

            return reply

    return "Hmm I don't know how to answer that. Type 'help' to see what I can do."


def console_chat():
    print("Chatbot ready. Type 'bye' to exit.\n")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        response = get_reply(user)
        print(f"Bot: {response}\n")
        if re.search(r'\b(bye|goodbye|exit|quit)\b', user.lower()):
            break


# ---- GUI version using tkinter ----
def gui_chat():
    import tkinter as tk
    from tkinter import scrolledtext

    def on_send(event=None):
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, tk.END)

        chat.config(state=tk.NORMAL)
        chat.insert(tk.END, f"You: {msg}\n", "user")

        reply = get_reply(msg)
        chat.insert(tk.END, f"Bot: {reply}\n\n", "bot")
        chat.config(state=tk.DISABLED)
        chat.see(tk.END)

        if re.search(r'\b(bye|goodbye|exit|quit)\b', msg.lower()):
            root.after(800, root.destroy)

    root = tk.Tk()
    root.title("Simple Chatbot")
    root.geometry("520x480")
    root.configure(bg="#2b2b2b")
    root.resizable(False, False)

    tk.Label(root, text="Chatbot", bg="#2b2b2b", fg="white",
             font=("Arial", 13, "bold")).pack(pady=8)

    chat = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD,
                                     width=58, height=22,
                                     bg="#1e1e1e", fg="white",
                                     font=("Courier", 10))
    chat.tag_config("user", foreground="#7eb8f7")
    chat.tag_config("bot",  foreground="#98e898")
    chat.pack(padx=10)

    # show a greeting when window opens
    chat.config(state=tk.NORMAL)
    chat.insert(tk.END, "Bot: Hey! Type 'help' to see what I can do.\n\n", "bot")
    chat.config(state=tk.DISABLED)

    bottom = tk.Frame(root, bg="#2b2b2b")
    bottom.pack(fill=tk.X, padx=10, pady=8)

    entry = tk.Entry(bottom, width=44, font=("Arial", 11),
                     bg="#3c3c3c", fg="white",
                     insertbackground="white", relief=tk.FLAT)
    entry.pack(side=tk.LEFT, ipady=5, padx=(0, 6))
    entry.bind("<Return>", on_send)
    entry.focus()

    tk.Button(bottom, text="Send", command=on_send,
              bg="#7eb8f7", fg="#1e1e1e",
              font=("Arial", 10, "bold"),
              relief=tk.FLAT, padx=10).pack(side=tk.LEFT)

    root.mainloop()


# ask user which mode they want
if __name__ == "__main__":
    print("1 - Console")
    print("2 - GUI window")
    choice = input("Pick one: ").strip()
    if choice == "2":
        gui_chat()
    else:
        console_chat()
