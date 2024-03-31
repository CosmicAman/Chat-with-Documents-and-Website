Chat with Documents and Websites 🤖📄

Welcome to Chat with Documents and Websites, a versatile Streamlit web application developed by Aman kumar for engaging in conversational interactions with documents (PDFs) and websites. Utilizing cutting-edge natural language processing (NLP) techniques, this project aims to provide users with an immersive and intuitive conversational experience. Whether you're exploring textual content from PDF files or engaging with dynamic web content, this application offers a seamless platform for interactive dialogue.

Table of Contents:

1. Features
2. Installation
3. Usage
4. Dependencies
5. Contributing
6. License
7. Contact

---

1. Features:

Multi-Modal Chat:
Switch effortlessly between two chat modes: "Chat with Website" and "Chat with PDFs", catering to diverse user preferences and requirements.

Document Interaction:
- PDF Mode: Upload PDF files and let the application extract text, facilitating seamless conversational interactions.
- Website Mode: Input a website URL, and dynamically fetch content to enable real-time conversations.

Advanced Text Processing:
- PDF Text Extraction: Accurately extract text from PDF files using the PyPDF2 library, ensuring a smooth user experience.
- Vector Store Creation: Construct a robust vector store from document chunks using langchain-core and langchain-community libraries, optimizing information retrieval and comprehension.

AI-powered Assistance:
- OpenAI Integration: Harness the power of OpenAI's GPT-3 model to provide intelligent responses to user queries, enhancing conversational depth and engagement.

Immersive Interface:
- Interactive Chat: Engage in immersive conversations through an intuitive chat interface, fostering a natural conversational flow.
- User-Friendly Experience: Ensure a seamless user experience with intuitive controls and clear visual feedback, enhancing ease of navigation and interaction.

---

2. Installation:

Clone the Repository:
git clone https://github.com/CosmicAman/Chat-with-Documents-and-Website

Navigate to the Project Directory:
cd your_repository

Install Dependencies:
pip install -r requirements.txt

---

3. Usage:

1. Run the Streamlit application:
streamlit run app.py

2. Access the application through the provided URL (typically http://localhost:8501).
3. Choose the desired chat mode (with a website or PDFs) and start engaging in immersive conversations with documents and websites.

---

4. Dependencies:

- Streamlit: A powerful framework for building and serving web applications with Python.
- python-dotenv: For loading environment variables from a .env file, ensuring secure configuration management.
- PyPDF2: A Python library for extracting text from PDF documents, facilitating seamless PDF interaction.
- langchain-core: Core functionalities for natural language processing, empowering advanced text processing capabilities.
- langchain-community: Community-contributed components for language processing, enhancing the project's versatility and functionality.
- langchain-openai: Integration with OpenAI's GPT-3 model for conversational AI, enabling intelligent responses and enhancing user engagement.

---

5. Contributing:

Contributions to this project are highly appreciated! Whether it's bug fixes, feature enhancements, or documentation improvements, your contributions help make this project better for everyone. Please feel free to open an issue or submit a pull request with your suggestions.

---

6. Contact:

For any inquiries, feedback, or collaboration opportunities, feel free to reach out to [aman7480nano@gmail.com].

---

This README provides comprehensive details about the project's features, installation steps, usage instructions, dependencies, contribution guidelines, licensing information, and contact details, tailored specifically for a personal project. Feel free to customize it further according to your project's specific needs and preferences.
