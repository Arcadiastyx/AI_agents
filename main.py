from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

model = Ollama(model= "llama3")

email = "The world will burn"


Classifier = Agent(
    role = "email claissifier",
    goal = "accurately classify emails based on their importance. Give every emain one of these ratings : important, casual, or spam",
    backstory = "You are an AI assistant whose only job is to classify emails accurately and honestly. Do not be afraid to give emails bad rating if they are not important. Your job is to help the user manage their inbox.",
    verbose = True,
    allow_delegetion = False,
    llm = model
)

Responder = Agent(
    role = "email responder",
    goal = "Based on the importance of the email, write a concise and simple response.if the email is rated <important> write a formal response, if the email is rated <casual> write a casual reponse, and if the email is rated <spam> ignore the email. No matter what, be very cocnise.",
    backstory = "You are an Ai assitant whose only job is to write short responses to emails based on their importance. The importance will be provided to you by the <classifier> agent.",
    verbose = True,
    allow_delegation = False,
    llm = model

)


Classify_email = Task(
    description = f" Classify the following email: '{email}'",
    agent = Classifier,
    expected_output = "One of these three options : 'important', 'casual', or 'spam'",
)


Respond_to_email = Task(
    description = f" Respond to the email: '{email}' based on the importance provided by the 'Classifier' agent.",
    agent = Responder,
    expected_output = "a very concise reponse to the email based on the importance provided by the 'Classifier' agent",
)

crew = Crew(
    agents = [Classifier, Responder],
    tasks= [Classify_email, Respond_to_email],
    verbose= 2,
    process= Process.sequential
)

output = crew.kickoff()
print(output)