"""
Chatting with GPT3.5 using openai API (by T.-W. Yoon, Mar. 2023)
"""

import openai
import os, clipboard


def openai_create(client, prompt, temperature=0.7, stream=True):
    """
    This function generates and print text based on the user's prompt.

    Args:
        client (client object): The client object from OpenAI.
        prompt (string): User prompt.
        temperature (float): Value between 0 and 1. Defaults to 0.7.

    Return:
        generated text.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=temperature,
        stream=stream
    )

    if stream:
        full_response = ""

        for chunk in response:
            delta = (chunk.choices[0].delta.content or "")
            print(delta, end="")
            full_response += delta
        print("\n")
        generated_text = full_response
    else:
        generated_text = response.choices[0].message.content
        print(generated_text)

    return generated_text


def chat_gpt():
    """
    This is a simple chatbot app with streaming, and
    the resulting conversation is saved to the clipboard.
    """

    # Set the API key and the client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    human_init = "\nHuman: "
    ai_init = "\nAI: "

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("\n************************",
          "\n* Chatting with GPT3.5 *",
          "\n************************",
          "\nSending 'bye' finishes the conversation.")

    while True:
        # Get the user's input
        user_input = input("\nHuman: ")

        # If the user enters "Bye", break out of the loop
        if user_input.lower() == "bye":
            break

        # Append the previous output and current input to the prompt
        prompt.append({"role": "user", "content": user_input})

        try:
            # Generate a response
            print(ai_init, end="")
            generated_text = openai_create(client, prompt, stream=True)
            prompt.append({"role": "assistant", "content": generated_text})
        except Exception as e:
            # If an exception occurs, generate an error message
            print(f"An error occurred: {e}")

    # Write the conversation to the clipboard
    to_clipboard = ""
    for message in prompt:
        if message['role'] != 'system':
            to_clipboard += f"\n{message['role']}: {message['content']}\n"
    clipboard.copy(to_clipboard)


if __name__ == "__main__":
    chat_gpt()
