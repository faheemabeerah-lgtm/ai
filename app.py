
import os
import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool

st.set_page_config(page_title="AI Research Agent", page_icon="🔎")
st.title("🔎 AI Research Agent")
st.write("Enter a topic to generate a research report.")

groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
serper_key = os.getenv("SERPER_API_KEY") or st.secrets.get("SERPER_API_KEY", "")

topic = st.text_input("Enter your research topic")

if st.button("Generate Report"):
    if not groq_key or not serper_key:
        st.error("Please configure GROQ_API_KEY and SERPER_API_KEY.")
    elif not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            llm = LLM(
                model="groq/openai/gpt-oss-120b",
                api_key=groq_key,
                temperature=0.2,
            )

            search_tool = SerperDevTool(api_key=serper_key)

            agent = Agent(
                role="AI Research Assistant",
                goal="Research topics and produce accurate, well-organized reports.",
                backstory="You are a careful researcher who verifies information and reports sources.",
                llm=llm,
                tools=[search_tool],
                allow_delegation=False,
                verbose=False,
            )

            task = Task(
                description=f"""
                Research the following topic: {topic}

                Use web search to gather information.
                Write a report with:
                1. Introduction
                2. Main findings and headings
                3. Examples and applications
                4. Challenges and limitations
                5. Conclusion
                6. Sources with URLs when available.

                Do not invent facts, citations, or URLs.
                """,
                expected_output="A detailed report with headings, conclusions, and source URLs.",
                agent=agent,
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )

            with st.spinner("Researching and writing your report..."):
                result = crew.kickoff()

            report = str(result)
            st.markdown(report)
            st.download_button(
                "Download report",
                data=report,
                file_name="research_report.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error("The report could not be generated.")
            st.code(f"{type(e).__name__}: {e}")
