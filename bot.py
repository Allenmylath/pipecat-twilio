import asyncio
import os
import sys
from loguru import logger
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import StartFrame, EndFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.logger import FrameLogger
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService, OpenAILLMContextFrame
from websocket_server import WebsocketServerParams, WebsocketServerTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from noisereduce_filter import NoisereduceFilter

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
firebase_creds = json.loads(os.environ.get('FIREBASE_CREDENTIALS', '{}'))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

class IntakeProcessor:
    def __init__(self, context: OpenAILLMContext):
        print(f"Initializing context from IntakeProcessor")
        context.add_message(
            {
                "role": "system",
                "content": "You are Jessica, an telephone call agent for a company called Tri-County Health Services. Your job is to collect important information from the user before their doctor visit. You're talking to Chad Bailey. You should address the user by their first name and be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. To insert pauses, insert “-” where you need the pause.Use two question marks to emphasize questions. For example, “Are you here??” vs. “Are you here?”Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous. Start by introducing yourself. Then, ask the user to confirm their identity by telling you their birthday, including the year. When they answer with their birthday, call the verify_birthday function.",
            }
        )
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "verify_birthday",
                        "description": "Use this function to verify the user has provided their correct birthday.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "birthday": {
                                    "type": "string",
                                    "description": "The user's birthdate, including the year. The user can provide it in any format, but convert it to YYYY-MM-DD format to call this function.",
                                }
                            },
                        },
                    },
                }
            ]
        )

    async def verify_birthday(
        self, function_name, tool_call_id, args, llm, context, result_callback
    ):
        if args["birthday"] == "1990-01-01":
            context.set_tools(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_prescriptions",
                            "description": "Once the user has provided a list of their prescription medications, call this function.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "prescriptions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "medication": {
                                                    "type": "string",
                                                    "description": "The medication's name",
                                                },
                                                "dosage": {
                                                    "type": "string",
                                                    "description": "The prescription's dosage",
                                                },
                                            },
                                        },
                                    }
                                },
                            },
                        },
                    }
                ]
            )
            # It's a bit weird to push this to the LLM, but it gets it into the pipeline
            # await llm.push_frame(sounds["ding2.wav"], FrameDirection.DOWNSTREAM)
            # We don't need the function call in the context, so just return a new
            # system message and let the framework re-prompt
            await result_callback(
                [
                    {
                        "role": "system",
                        "content": "Next, thank the user for confirming their identity, then ask the user to list their current prescriptions. Each prescription needs to have a medication name and a dosage. Do not call the list_prescriptions function with any unknown dosages.",
                    }
                ]
            )
        else:
            # The user provided an incorrect birthday; ask them to try again
            await result_callback(
                [
                    {
                        "role": "system",
                        "content": "The user provided an incorrect birthday. Ask them for their birthday again. When they answer, call the verify_birthday function.",
                    }
                ]
            )

    async def start_prescriptions(self, function_name, llm, context):
        print(f"!!! doing start prescriptions")
        # Move on to allergies
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_allergies",
                        "description": "Once the user has provided a list of their allergies, call this function.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "allergies": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "What the user is allergic to",
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                }
            ]
        )
        context.add_message(
            {
                "role": "system",
                "content": "Next, ask the user if they have any allergies. Once they have listed their allergies or confirmed they don't have any, call the list_allergies function.",
            }
        )
        print(f"!!! about to await llm process frame in start prescrpitions")
        await llm.process_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)
        print(f"!!! past await process frame in start prescriptions")

    async def start_allergies(self, function_name, llm, context):
        print("!!! doing start allergies")
        # Move on to conditions
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_conditions",
                        "description": "Once the user has provided a list of their medical conditions, call this function.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "The user's medical condition",
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                },
            ]
        )
        context.add_message(
            {
                "role": "system",
                "content": "Now ask the user if they have any medical conditions the doctor should know about. Once they've answered the question, call the list_conditions function.",
            }
        )
        await llm.process_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)

    async def start_conditions(self, function_name, llm, context):
        print("!!! doing start conditions")
        # Move on to visit reasons
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_visit_reasons",
                        "description": "Once the user has provided a list of the reasons they are visiting a doctor today, call this function.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "visit_reasons": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "The user's reason for visiting the doctor",
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                }
            ]
        )
        context.add_message(
            {
                "role": "system",
                "content": "Finally, ask the user the reason for their doctor visit today. Once they answer, call the list_visit_reasons function.",
            }
        )
        await llm.process_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)

    async def start_visit_reasons(self, function_name, llm, context):
        print("!!! doing start visit reasons")
        # move to finish call
        context.set_tools([])
        context.add_message(
            {"role": "system", "content": "Now, thank the user and end the conversation."}
        )
        await llm.process_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)    
        
    async def save_data(self, function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Saving data: {args}")
        
        # Get the user document reference
        user_ref = db.collection('users').document('chad_bailey')  # You might want to make this dynamic

        # Update the user document based on the function name
        if function_name == "list_prescriptions":
            user_ref.update({"prescriptions": args["prescriptions"]})
        elif function_name == "list_allergies":
            user_ref.update({"allergies": args["allergies"]})
        elif function_name == "list_conditions":
            user_ref.update({"conditions": args["conditions"]})
        elif function_name == "list_visit_reasons":
            user_ref.update({"visit_reasons": args["visit_reasons"]})

        logger.info(f"Data saved to Firebase for function: {function_name}")
        await result_callback(None)

async def main():
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            host="",
            port=int(os.environ["PORT"]),
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            audio_in_filter=NoisereduceFilter(),
        )
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30",  # British Lady
    )

    messages = []
    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)
    intake = IntakeProcessor(context)

    # Register functions
    llm.register_function("verify_birthday", intake.verify_birthday)
    llm.register_function(
        "list_prescriptions",
        intake.save_data,
        start_callback=intake.start_prescriptions
    )
    llm.register_function(
        "list_allergies",
        intake.save_data,
        start_callback=intake.start_allergies
    )
    llm.register_function(
        "list_conditions",
        intake.save_data,
        start_callback=intake.start_conditions
    )
    llm.register_function(
        "list_visit_reasons",
        intake.save_data,
        start_callback=intake.start_visit_reasons
    )

    fl = FrameLogger("LLM Output")

    pipeline = Pipeline([
        transport.input(),  # WebSocket input
        stt,  # Speech-To-Text
        context_aggregator.user(),  # User responses
        llm,  # LLM
        fl,  # Frame logger
        tts,  # Text-To-Speech
        transport.output(),  # WebSocket output
        context_aggregator.assistant(),  # Assistant responses
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Clear existing messages and create new context
        context.messages.clear()
               
        # Reinitialize IntakeProcessor with fresh context
        intake = IntakeProcessor(context)
        
        # Reset context aggregator with fresh context
        context_aggregator = llm.create_context_aggregator(context)
        
        print(f"New client connected. Fresh context created: {context}")
        await task.queue_frames([OpenAILLMContextFrame(context)])
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
