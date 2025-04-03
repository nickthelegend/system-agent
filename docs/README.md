<p align='center'>
   <img style="display: block; margin: 0 auto;" src='logo.png'/>
</p>

This project aims to replicate how a person naturally interacts with a computer to solve tasks. The system utilizes multiple agents to handle tasks such as web browsing, terminal operations, GUI interactions, and memory storage for adaptive decision-making. The **Memory Agent** is pivotal in recalling previous actions, which enhances the efficiency and accuracy of solving related tasks in the future.

### Demo

https://github.com/user-attachments/assets/63b21b86-a829-4e30-a2c4-cca33e0ba907

### Architecture

The following diagram outlines the high-level interactions between the different agents:

![Image of the architecture](diagram.svg)

### Development Status

This project is currently in the **development stage**, but significant progress has been made on its core components:

- **Web Search Agent**: Fully functional and effectively automates web interactions to retrieve data.
- **System Agent**: Almost fully functional, capable of handling GUI tasks. Currently supported only on Windows operating systems.
- **Terminal Agent**: Working at an optimal level for executing advanced or specific terminal commands.
- **Memory Agent**: Still under development. Thinking whether a centralized memory agent or dedicated memory for each agent would provide better efficiency and functionality.

### Important Note

This program runs directly on your machine and leverages a large language model (LLM) to execute commands. **Use this system with caution**, as it can interact with your system files, settings, and applications. Improper use or unintended commands may lead to potential harm to your system. 

## Project Components

The system is structured around the following agents, each with a specialized role:

1. **System Agent**  
   Handles GUI-based tasks such as opening files, manipulating windows, and interacting with desktop elements. Currently supported only on Windows and nearing full functionality.

2. **Web Search Agent**  
   Automates web browsing to gather data or solve tasks using online resources. This agent is fully functional and reliable.

3. **Terminal Agent**  
   Executes terminal commands for advanced operations. This component is fully operational and performs tasks efficiently.

4. **Memory Agent**  
   Stores and recalls past actions to improve decision-making and avoid redundant operations. Still under development, with ongoing discussions about the best architecture for memory management.

## Workflow

1. The **Memory Agent** (or agent-specific memory) stores methodologies and steps used for solving tasks.
2. When a new query is received, the system checks past stored actions to replicate or adapt a similar solution.
3. The system integrates the **System**, **Web Search**, and **Terminal Agent** as needed to solve the task, mimicking a human approach.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/jeomon/computer-agent.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the agent system:
   ```bash
   python app.py
   Enter your query: <your query goes here>
   ```

## Example

## Find the weather of a location and write in notepad.

```plaintext
Entering Computer Agent
Thought: First, I need to get the weather information for Singapore. I can use the Web Agent for this task. Once I have the weather information, I'll use the System Agent to open Notepad and write the information into it.
Agent Name: Web Agent
Agent Request: Search for the current weather in Singapore and provide the information.
Entering Web Agent
Thought: I need to search for the current weather in Singapore. I will use the search engine to find this information. I will start by going to google.com.
Action Name: GoTo Tool
Action Input: {'url': 'https://www.google.com'}
Observation: Navigated to https://www.google.com
Thought: I'm on the Google homepage. I need to type "current weather in Singapore" in the search box and then click the Google Search button.
Action Name: Type Tool
Action Input: {'index': 6, 'text': 'current weather in Singapore'}
Observation: Typed current weather in Singapore in element 6
Thought: I have typed the query in the search box. Now I need to click on the Google Search button to get the results.  
Action Name: Click Tool
Action Input: {'index': 21}
Observation: Clicked element 21
Thought: I have successfully searched for the current weather in Singapore. The current temperature is 28 degrees Celsius, with light rain. The precipitation is 45%, humidity is 75%, and wind is 13 km/h. I can provide this information to the user.
Final Answer: The current weather in Singapore is 28 degrees Celsius with light rain. The precipitation is 45%, humidity is 75%, and wind is 13 km/h.
Agent Response: The current weather in Singapore is 28 degrees Celsius with light rain. The precipitation is 45%, humidity is 75%, and wind is 13 km/h.
Thought: Now that I have the weather information, I need to open Notepad and write the weather information into it. I will use the System Agent for this task.
Agent Name: System Agent
Agent Request: Open Notepad. Then type the following: "The current weather in Singapore is 28 degrees Celsius with light rain. The precipitation is 45%, humidity is 75%, and wind is 13 km/h."
Entering System Agent
Thought: First, I need to open Notepad. I will use the search bar to find and open it.
Action Name: Click Tool
Action Input: {'index': 2, 'button': 'left'}
Final Answer: The text "The current weather in Singapore is 28 degrees Celsius with light rain. The precipitation is 45%, humidity is 75%, and wind is 13 km/h." has been typed into Notepad.            
Agent Response: The text "The current weather in Singapore is 28 degrees Celsius with light rain. The precipitation is 45%, humidity is 75%, and wind is 13 km/h." has been typed into Notepad.
Thought: I have successfully retrieved the weather information using the Web Agent and then written it into Notepad using the System Agent. The task is complete.
Final Answer: The weather information for Singapore has been written into notepad. The current weather in Singapore is 28 degrees Celsius with light rain. The precipitation is 45%, humidity is 75%, and wind is 13 km/h
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

### What this means:

- ✔️ You can view and use this code for personal and educational purposes
- ✔️ You can modify the code
- ✔️ You must preserve the copyright and license notices
- ✔️ You must disclose the source code when distributing the software
- ✔️ Changes must be released under the same license
- ❌ You cannot use this code for commercial purposes without explicit permission
- ❌ No warranty is provided

For more information about the AGPL-3.0 license, please visit: https://www.gnu.org/licenses/agpl-3.0.en.html

## Future Directions

- Enhance the **System Agent** to handle more advanced and complex GUI tasks.
- Finalize the design and implementation of the **Memory Agent**, determining whether centralized or agent-specific memory is most effective.
- Continue refining the **Web Search Agent** and **Terminal Agent** for robust performance across a wide range of scenarios.

## **Contact**

For queries or support, please reach out via GitHub Issues.

E-mail: jeogeoalukka@gmail.com
---
