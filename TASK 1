# Rule-based chatbot with simple conversation flow

# Function to handle chatbot responses
def chatbot_response(user_input):
    # Convert the user's input to lowercase for easy comparison
    user_input = user_input.lower()

    # Greeting rule
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you today?"
    
    # Asking about the chatbot's name
    elif "your name" in user_input:
        return "I'm a simple rule-based chatbot. You can call me 'ChatBot'."
    
    # Asking about the time
    elif "time" in user_input:
        return "I'm not equipped with a clock, but I can help with other things!"
    
    # Asking for help
    elif "help" in user_input:
        return "Sure, I’m here to help. Ask me anything you like!"
    
    # Rule for generic goodbyes
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"
    
    # Rule for questions about the weather
    elif "weather" in user_input:
        return "I can't check the weather right now, but you could try a weather app!"
    
    # Rule for unknown inputs
    else:
        return "Sorry, I don't understand that. Could you please rephrase?"

# Main loop for user interaction
def main():
    # Print a welcome message
    print("ChatBot: Hello! Type something to start chatting (or type 'bye' to exit).")

    # Keep the conversation running until the user decides to quit
    while True:
        # Get input from the user
        user_input = input("You: ")

        # If the user types 'bye', exit the loop
        if "bye" in user_input.lower():
            print("ChatBot: Goodbye!")
            break

        # Get the chatbot's response
        response = chatbot_response(user_input)

        # Print the chatbot's response
        print(f"ChatBot: {response}")

# Start the chatbot when the script runs
if __name__ == "__main__":
    main()
