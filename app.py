import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

st.title("PHIDATA VIDEO SUMMARIZER AI AGENT")
st.header("Using Gemini")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video Summarizer AI",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

agent=initialize_agent()

video_file=st.file_uploader(
    "Upload a video file",
    type=["mp4","avi"],
    help="Upload the file here for AI analysis"
)


if video_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path=temp_video.name
    st.video(video_path,format='video/mp4',start_time=0)

    user_query=st.text_area(
        "What insights are you seeking from th video?",
        placeholder="Ask anything about the video content",
        help="Provide questions on what you want to know"
    )

    if st.button("Analyze the video",key="analyze_video_button"):
        if not user_query:
            st.warning("Please give question to analyze")
        else:
            try:
                with st.spinner("Processing videos and gathering insights..."):
                    processed_video=upload_file(video_path)
                    while processed_video.state.name=="PROCESSING":
                        time.sleep(1)
                        processed_video=get_file(processed_video.name)
                    
                    analysis_prompt=(
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights.

                        Query:{user_query}
                        
                        Provide detailed, user friendly and actionable response."""
                    )

                    response=agent.run(analysis_prompt,videos=[processed_video])

                st.subheader("Analysis result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"Error occured during analysis: {error}")

            finally:
                Path(video_path).unlink(missing_ok=True)
