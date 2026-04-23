import streamlit as st
import PyPDF2
import os
import io
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="AI Resume Critiquer", page_icon=":guardsman:", layout="centered")

st.title("AI resume Critiquer")
st.markdown("Upload Your resume to get AI-powered feedback tailored to your needs")

uploaded_file = st.file_uploader("Upload Your resume (PDF OR TXT)",type= ["pdf","txt"])
job_role = st.text_input("Enter the Job role you're targetting(Optional)",placeholder="e.g Data Scientist, Software Engineer, etc.")

def extract_text_from_pdf(pdf_file):
    pdf_read = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_read.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(file.read()))
    return file.read().decode("utf-8")


analyze = st.button("Analyze Resume")
if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)

        if not file_content.strip():
            st.error("File does not have any content")
            st.stop()
        prompt = """
        You are an AI reume critiquer Please Provide 
        feedback on the following resume content based on
        1. Content Clarity and impact
        2. Skills Presentation
        3. Experience Relevance
        4. Specific Improvements for {job_role} 

        Resume Content:
        {file_content}

Once you've provided the fedback give a rating from 1 to 10 based on the overall quality of the resume.
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        llm = OllamaLLM(model="llama3.2")
        with st.spinner("Loading..."):
            chain = prompt_template | llm
            response = chain.invoke({"file_content":file_content,"job_role":job_role if job_role else "General Job Application"})
            st.markdown("### AI Feedback")
            st.markdown(response)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
    else:
        st.success("Resume analyzed successfully!")
else:
    st.warning("Please upload a resume file and click 'Analyze Resume' to get feedback.")

        

"""
        Feedback

Content Clarity and Impact (7/10)

The resume is well-structured, and the candidate has clearly outlined their professional summary. However, some sections could be rephrased for better impact. For instance, the "Skills" section feels a bit lengthy and disorganized. Some skills are listed under categories without clear explanations, making it difficult to gauge the level of proficiency.

To improve, consider breaking down long lists into concise bullet points with brief descriptions. Additionally, use specific examples from experiences or projects to demonstrate the application of each skill.

Skills Presentation (6/10)

The skills section is overwhelming due to its extensive length and lack of organization. Some skills are not relevant to software engineering, such as Adobe Suite, Microsoft Office, and certain tools like Docker, CI/CD, Cloud Deployment, and GPU Inference (although these are listed with a "Conceptual" label).

Focus on highlighting the most relevant technical skills for a software engineer position. Consider using a more organized format, such as:

Programming Languages: Python, JavaScript
Web Frameworks/Libraries: Django, Flask, React.js, Next.js
AI/ML: LLMs (Mistral, GPT-4V/Gemini), AI Agents, PyTorch
Experience Relevance (8/10)

The candidate has diverse experience in logistics, customer service, and software engineering. However, some details feel like they belong more in a cover letter or LinkedIn profile.

To improve, focus on showcasing the skills directly related to the job requirements. For instance, instead of mentioning "complex logistics operations," highlight "project management" and "collaboration with international teams."

Specific Improvements for Software Engineer

Emphasize your proficiency in programming languages like Python and JavaScript.
Highlight your experience with web frameworks/Libraries, such as Django and Flask.
Tailor your AI/ML skills to show relevance to software engineering, e.g., "Natural Language Processing" or "Machine Learning for Data Analysis."
Remove irrelevant tools and focus on the most relevant technical skills.
Additional Suggestions

Use action verbs consistently throughout the resume (e.g., "Developed," "Implemented," "Collaborated").
Quantify achievements, such as "Improved efficiency by 30% through optimized logistics operations."
Consider adding a personal project or contribution to open-source projects to demonstrate your passion and skills.
Rating: 7.5/10

Overall, the resume has potential but could benefit from refinement in terms of organization, clarity, and relevance to software engineering positions. By focusing on technical skills, experience, and action verbs, you can enhance its overall impact and increase your chances of standing out in a competitive job market.
        """