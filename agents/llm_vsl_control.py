import openai
import random
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
os.environ["OPENAI_API_BASE"] = "https://openai.api2d.net/v1"
os.environ["OPENAI_API_KEY"] = ""


class llmvslagent:
    def __init__(self):
        self.llm = OpenAI(model_name="gpt-3.5-turbo")
        template = """{question}\n\n"""
        self.prompt = PromptTemplate(template=template, input_variables=["question"])
    
    def generate_chat_completion(self, question):
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        return llm_chain.run(question)
        
    def generate_decision(self, flow_sensor,occupancy):
        # description = '''

        # '''
        input_=f"""
        You are a speed limit and ramp metering controller. You are designed to make decisions based on the merge scenario information provided by humans.
        In the following, you will be given some observations from the sensors of the loop detectors.
        You only need to make inferences based on the available information.
        You need to analyze the scenario step by step.
        You should NOT assume scenarios that are not happening but only for the current observation.
        The section 9832 is the main entrance lane, it has three lanes, 181.4 meters long; The section 9712 is the controlled bottleneck, 359 meters long, it has four lanes; the ramp section 9813 is the merge lane, 44 meters long, only 1 lane; the section 9728 is the outer section, connecting the end of 9712, has 3 lanes.
        Here is the observation from the sensors: ```{flow_sensor}```
        Here is the occupancy measurement of the controlled section 9712: ```{occupancy}```
        Your decision is to decide the speed limit(maximum is 30m/s,only generate value no unit, e.g., 30) for each lane on section 9712 and whether to keep open or close ramp section 9813(only genrate value 0 or 1).
        Generate your decision in JSON format:
        '''
        '9712_lane_0_speed_limit': 'speedlimit0',
        '9712_lane_1_speed_limit': 'speedlimit1',
        '9712_lane_2_speed_limit': 'speedlimit2',
        '9712_lane_3_speed_limit': 'speedlimit3',
        'decision_of_ramp': 'open:1, close: 0',
        'explain': "a brief explanation"
        """
        print('input prompt',input_)

        output=self.generate_chat_completion(str(input_))

        return output
