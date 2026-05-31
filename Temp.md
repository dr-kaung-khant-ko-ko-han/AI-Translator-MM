CrewAI နဲ့ LangGraph က AI agent တွေ၊ workflow တွေ တည်ဆောက်ဖို့ သုံးတဲ့ Python framework နှစ်ခုပါ။ အောက်မှာ တစ်ခုချင်းစီကို ဘယ်လိုစသုံးရမလဲ၊ ဥပမာ code တွေနဲ့ ရှင်းပြပေးမယ်နော်။

---

## 1. CrewAI – Agent အဖွဲ့တွေနဲ့ အလုပ်ခွဲလုပ်ခြင်း

CrewAI က **Role-based agent** တွေကို သတ်မှတ်ပြီး သူတို့ကို task တွေခွဲပေးကာ အတူတူပူးပေါင်းလုပ်ဆောင်စေတဲ့ framework ပါ။ ရိုးရိုးရှင်းရှင်း AI "crew" တစ်ခု ဖွဲ့ပြီး အလုပ်တွေကို အလိုအလျောက် ပြီးမြောက်အောင် လုပ်နိုင်တယ်။

### ထည့်သွင်းခြင်း
```bash
pip install crewai crewai-tools
```

### အခြေခံအဆင့်များ
1. **Agent** – ဘယ်သူလဲ၊ ဘာလုပ်မလဲ၊ ဘယ်လို စဉ်းစားမလဲ (role, goal, backstory, llm) သတ်မှတ်ပါ။
2. **Task** – Agent တစ်ဦးချင်းစီအတွက် လုပ်ရမယ့်အလုပ် (description, expected_output)။
3. **Crew** – Agents နဲ့ tasks တွေကိုစုစည်း၊ process (sequential သို့မဟုတ် hierarchical) သတ်မှတ်ပြီး `kickoff()` လုပ်ပါ။

### ဥပမာ – သုတေသနအဖွဲ့တစ်ဖွဲ့
```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 1. LLM သတ်မှတ် (OpenAI API key လိုမယ်)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. Agents များ
researcher = Agent(
    role="Senior Researcher",
    goal="Find the latest developments in AI",
    backstory="You are an expert researcher with a knack for finding cutting-edge info.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Tech Writer",
    goal="Write a clear 3-paragraph summary about the findings",
    backstory="You turn complex tech info into simple, engaging content.",
    llm=llm,
    verbose=True
)

# 3. Tasks များ
research_task = Task(
    description="Search for the top 3 AI breakthroughs in 2025 so far.",
    expected_output="A bullet list of breakthroughs with brief explanations.",
    agent=researcher
)

write_task = Task(
    description="Using the research, write a 3-paragraph summary for a blog post.",
    expected_output="3-paragraph markdown text ready to publish.",
    agent=writer
)

# 4. Crew ဖွဲ့ပြီး စတင်
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # တစ်ခုပြီးမှတစ်ခု
    verbose=True
)

result = crew.kickoff()
print(result)
```

**ထူးခြားချက်** – CrewAI မှာ agent တစ်ဦးချင်းစီမှာ ကိုယ်ပိုင် memory၊ role-based prompting ရှိပြီး အလိုအလျောက် ပူးပေါင်းဆောင်ရွက်သွားတယ်။

---

## 2. LangGraph – Stateful Graph တွေနဲ့ Agent Workflow

LangGraph က **graph-based state machine** တစ်ခုလို အလုပ်လုပ်ပြီး AI workflow တွေ (agents, multi-step chains) ကို လမ်းကြောင်းအဆင့်ဆင့်ခွဲကာ ထိန်းချုပ်နိုင်တယ်။ Agent တစ်ခုရဲ့ ဆုံးဖြတ်ချက်အလိုက် နောက်တစ်ဆင့်ကို လမ်းလွှဲနိုင်တဲ့ စွမ်းရည်ရှိတယ်။

### ထည့်သွင်းခြင်း
```bash
pip install langgraph langchain-openai
```

### အခြေခံသဘော
- **State** – Workflow တစ်လျှောက် သယ်ဆောင်သွားမယ့် data structure (TypedDict သုံးလေ့ရှိ)။
- **Nodes** – Python function တွေ၊ state ကို input ယူပြီး update လုပ်တယ်။
- **Edges** – Node တစ်ခုပြီးရင် ဘယ် node ဆက်လုပ်မလဲဆိုတာ သတ်မှတ် (conditional edge လည်းရှိ)။
- **Compile** – Graph ကို compile လုပ်ပြီး `invoke()` ခေါ်သုံးတယ်။

### ဥပမာ – ရိုးရှင်းတဲ့ Agent with Tool
LangGraph နဲ့ အမေးအဖြေ + search tool ခေါ်မယ့် agent တစ်ခုတည်ဆောက်ပါမယ်။

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# Tool သတ်မှတ် (Tavily API key လိုမယ်)
search = TavilySearchResults(max_results=1)

# 1. State
class AgentState(TypedDict):
    messages: list
    next_step: str

# 2. Nodes
def chatbot(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def tool_node(state: AgentState):
    last_msg = state["messages"][-1]
    # LLM က tool ခေါ်ရန် တောင်းဆိုလာရင် (ဒီနေရာတွင် အရိုးရှင်းဆုံးပုံစံနဲ့ ပြထားတာ)
    if "search" in last_msg.content.lower():
        result = search.invoke(last_msg.content)
        return {"messages": state["messages"] + [HumanMessage(content=str(result))]}
    else:
        return {"messages": state["messages"]}

# 3. Route လုပ်မယ့် function
def router(state: AgentState) -> Literal["tool", "end"]:
    last_msg = state["messages"[-1]
    if "search" in last_msg.content.lower():
        return "tool"
    else:
        return "end"

# 4. Graph တည်ဆောက်
builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tool", tool_node)

builder.set_entry_point("chatbot")
builder.add_conditional_edges("chatbot", router, {"tool": "tool", "end": END})
builder.add_edge("tool", "chatbot")  # tool ပြီးရင် chatbot ပြန်သွားမယ်

graph = builder.compile()

# 5. Run
user_input = "Search for the weather in Yangon today"
state = graph.invoke({"messages": [HumanMessage(content=user_input)]})
for msg in state["messages"]:
    print(msg.content)
```

**အားသာချက်** – LangGraph က အဆင့်တိုင်းကို ကိုယ်တိုင်ထိန်းချုပ်နိုင်ပြီး ရှုပ်ထွေးတဲ့ looping, branching, human-in-the-loop တွေကို လွယ်လွယ်ကူကူတည်ဆောက်နိုင်တယ်။

---

## CrewAI vs LangGraph – ဘယ်ဟာကို ဘယ်အချိန်သုံးမလဲ?

| **CrewAI** | **LangGraph** |
|------------|---------------|
| Multi-agent ပူးပေါင်းဆောင်ရွက်မှုကို အဓိကထား | Stateful graph flow ကို အသေးစိတ်ထိန်းချုပ်ချင်ရင် |
| Role-based prompting တွေ၊ autonomous delegation | ကိုယ်ပိုင် logic, tools, loops တွေ လွတ်လွတ်လပ်လပ်ရေးချင်ရင် |
| ရိုးရှင်းတဲ့ သုတေသန၊ အကြောင်းအရာရေး အဖွဲ့ | RAG, multi-step reasoning, human approval step ထည့်ချင်ရင် |
| ချက်ချင်းစမ်းကြည့်လို့ကောင်း (low-code ဆန်ဆန်) | ပိုမို flexible၊ granular control ရှိ |

နှစ်ခုကို တွဲသုံးလို့လည်းရပါတယ် – ဥပမာ LangGraph ထဲမှာ CrewAI crew တစ်ခုကို node တစ်ခုအနေနဲ့ ထည့်ပြီး အလုပ်လုပ်စေတာမျိုး။

---

**နောက်ထပ်လေ့လာရန်**
- CrewAI docs: [https://docs.crewai.com](https://docs.crewai.com)
- LangGraph docs: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)

မေးခွန်းရှိရင် ဆက်မေးပါခင်ဗျာ။ ကိုယ်လုပ်ချင်တဲ့ project အတွက် ဘယ်ဟာပိုသင့်တော်လဲဆိုတာ တိုင်ပင်လို့ရပါတယ်။

---

DeepSeek TUI (Terminal User Interface) ဆိုတာ **Terminal (Command Line) ထဲကနေ DeepSeek AI model နဲ့ တိုက်ရိုက်စကားပြောနိုင်အောင် ပြုလုပ်ထားတဲ့ ဆော့ဖ်ဝဲတစ်ခု** ပါ။  
ဘယ်လို web browser မှမလိုဘဲ၊ mouse မသုံးဘဲ keyboard နဲ့တင် DeepSeek ရဲ့ စွမ်းဆောင်ရည်တွေကို အပြည့်အဝခံစားနိုင်ပါတယ်။

---

## ဘာတွေထူးခြားလဲ၊ ဘာတွေလုပ်ပေးနိုင်လဲ?

- **Terminal အတွင်းမှာပဲ အဆင်ပြေပြေသုံးလို့ရတယ်** – GUI မရှိဘဲ developer တွေ၊ server admin တွေ အနေနဲ့ မိမိတို့ပတ်ဝန်းကျင်မှာတင် AI ကို အကူအညီတောင်းနိုင်တယ်။
- **Streaming စနစ်** – စာသားတွေကို အချိန်နဲ့တပြေးညီ တန်းစီထွက်လာတယ် (typewriter effect)၊ ချက်ချင်းဖတ်လို့ရတယ်။
- **Markdown Rendering** – code block တွေ၊ ဇယားတွေ၊ bold/italic စာသားတွေကို terminal မှာ သပ်သပ်ရပ်ရပ် ပြပေးတယ်။
- **Conversation History** – စကားပြောထားတဲ့ မှတ်တမ်းကို session တစ်ခုအတွင်း မှတ်သားထားတယ်။ တချို့ TUI မှာ history ကို ဖိုင်နဲ့သိမ်းတာတွေလည်းရှိတယ်။
- **Configurable** – API key, model version, temperature စတဲ့ setting တွေကို config ဖိုင် (သို့) environment variable ကနေ ပြင်လို့ရတယ်။

---

## ဘယ်လိုအလုပ်လုပ်လဲ?

DeepSeek TUI ရဲ့ အတွင်းပိုင်း လုပ်ဆောင်ချက်ကို အကျဉ်းချုပ်ရရင် –

1. **User Input** – Terminal ကနေ စာရိုက်ထည့်တယ် (prompt_toolkit, textual စတဲ့ library တွေသုံးပြီး multi-line, syntax highlighting ပံ့ပိုး)။
2. **API Request** – ရိုက်ထည့်လိုက်တဲ့စာကို DeepSeek API server ဆီ HTTP request (အများအားဖြင့် OpenAI-compatible `/v1/chat/completions`) အဖြစ် ပို့ပေးတယ်။
3. **Streaming Response** – ပြန်လာတဲ့ response ကို chunk အလိုက် လက်ခံပြီး terminal ပေါ် တစ်ဆင့်ချင်းပြပေးတယ်။
4. **Markdown Parsing** – Code block လို နေရာတွေမှာ ရောင်စုံ syntax highlight (rich လို library သုံးပြီး) တို့နဲ့ ပြပေးတယ်။
5. **State ထိန်းသိမ်းခြင်း** – Conversation တစ်ခုလုံးရဲ့ messages တွေကို list အဖြစ် memory ထဲမှာသိမ်းထားပြီး နောက် request တိုင်းမှာ context အဖြစ် ပြန်ထည့်ပေးတယ်။

---

## သုံးစွဲနည်း အကျဉ်း

### ထည့်သွင်းခြင်း (ဥပမာ Python package အနေနဲ့)
```bash
pip install deepseek-tui
# သို့မဟုတ် GitHub မှ clone လုပ်ပြီးသုံး
```

### API Key ထည့်ခြင်း
```bash
export DEEPSEEK_API_KEY="sk-your-api-key"
```

### စတင်အသုံးပြုခြင်း
```bash
deepseek-tui
```
Terminal ထဲမှာ `>>>` prompt ပေါ်လာရင် စာရိုက်ပြီး Enter ခေါက်ရုံပါပဲ။ ထွက်ချင်ရင် `/exit` (သို့) `Ctrl+C` နှိပ်ပါ။

---

## ဘယ်သူတွေအတွက်အကျိုးရှိလဲ?

- **Developer** – Code ရေးနေရင်း IDE မပြောင်းဘဲ terminal ကနေတစ်ဆင့် AI ကိုမေးမြန်းနိုင်။
- **System Admin** – SSH နဲ့ server ထဲဝင်ပြီး troubleshooting လုပ်တဲ့အခါ deepseek-tui ကို တွဲသုံးနိုင်။
- **CLI Lover** – ဘာမဆို keyboard နဲ့ပဲ အလုပ်လုပ်ချင်သူများအတွက် အကောင်းဆုံး tool တစ်ခု။

ဒါဟာ DeepSeek ရဲ့ မိသားစုဝင် tool တစ်ခုဖြစ်ပြီး terminal ကနေ AI နဲ့အဆင်ပြေစွာ ဆက်သွယ်နိုင်ဖို့ တည်ဆောက်ထားတာပါ။ တရားဝင် GitHub repo နဲ့ နောက်ဆုံးထွက် ဗားရှင်းတွေကို DeepSeek ရဲ့ ထုတ်ပြန်ချက်တွေကနေ စောင့်ကြည့်နိုင်ပါတယ်။

နောက်ထပ်သိချင်တဲ့အကြောင်းအရာရှိရင်လည်း ဆက်မေးလို့ရပါတယ်။ 😊

---

ကွန်ပျူတာထဲက ဖိုင်တွေ (စာသားဖိုင်တွေ) ကို မြန်မာဘာသာပြန်လို့ရပါတယ်။ ဒါပေမယ့် ဖိုင်တစ်ခုလုံးကို တစ်ခါတည်း အလိုအလျောက်ပြန်ဖို့ဆိုရင် **နည်းလမ်းကို ရွေးချယ်ရပါမယ်**။ DeepSeek မော်ဒယ်ကို သုံးပြီးပြန်ချင်တာလား၊ တခြား အခမဲ့/လွယ်ကူတဲ့ ကိရိယာတွေ သုံးချင်တာလား ဆိုတဲ့အပေါ် မူတည်ပါတယ်။

အောက်မှာ အသုံးဝင်မယ့် နည်းလမ်း (၃) ခုကို ပြောပြပေးပါမယ်။


## 1. DeepSeek ကို Chat/Web မှာတင် လက်နဲ့ ကူးထည့်ပြန်ခြင်း (အလွယ်ဆုံး)

ဖိုင်ထဲက စာသားကို select လုပ်ပြီး Copy ကူး → DeepSeek Web Chat (သို့) DeepSeek TUI မှာ  
> "အောက်ပါစာသားကို မြန်မာလိုပြန်ပေးပါ: [စာသား]"

လို့ရိုက်ထည့်ပြီး ဘာသာပြန်ခိုင်းလို့ရပါတယ်။ ဖိုင်ငယ်တွေ၊ အမြန်ပြန်ချင်တဲ့အခါ အဆင်ပြေပါတယ်။


## 2. Python Script ရေးပြီး DeepSeek API နဲ့ အလိုအလျောက်ပြန်ခြင်း

DeepSeek API (OpenAI-compatible) ကိုသုံးပြီး `.txt`, `.md`, `.srt` စတဲ့ ဖိုင်တွေကို တစ်ပြိုင်နက် ဘာသာပြန်နိုင်ပါတယ်။

**လိုအပ်တာများ**
- Python 3.8+
- DeepSeek API Key ([ဒီမှာရယူပါ](https://platform.deepseek.com/))
- `openai` Python library

### Script နမူနာ (ဖိုင်တစ်ခုလုံးကို မြန်မာလိုပြန်)

```python
import openai

# DeepSeek API သတ်မှတ်
client = openai.OpenAI(
    api_key="sk-your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)

def translate_file(input_path, output_path, source_lang="English", target_lang="Burmese"):
    # ဖိုင်ဖတ်
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # စာသားကို တစ်ခါတည်းပို့မယ် (ဖိုင်မကြီးရင်)
    response = client.chat.completions.create(
        model="deepseek-chat",  # ဒါမှမဟုတ် deepseek-reasoner
        messages=[
            {"role": "system", "content": f"You are a professional translator. Translate the following {source_lang} text into {target_lang}. Preserve formatting and line breaks."},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        stream=False
    )
    
    translated = response.choices[0].message.content
    
    # ဘာသာပြန်ပြီးသားကို ဖိုင်အသစ်မှာ သိမ်း
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
    
    print(f"ဘာသာပြန်ပြီးပါပြီ -> {output_path}")

# သုံးမယ်
translate_file("english_note.txt", "myanmar_note.txt")
```

> **မှတ်ချက်** – ဖိုင်အရမ်းကြီးရင် token limit ကျော်နိုင်တာမို့ စာပိုဒ်ခွဲပြီး ပို့တာမျိုး လုပ်ဖို့လိုပါမယ်။ (ကျွန်တော် ကူညီပေးနိုင်ပါတယ်)


## 3. GUI/Offline/Free Tools သုံးခြင်း (DeepSeek မပါ)

အွန်လိုင်းမသုံးချင်ဘူး (သို့) အခမဲ့ ကိရိယာသုံးချင်ရင် –  

| ကိရိယာ | အားသာချက် |
|--------|------------|
| **Google Translate (Documents)** | `.docx`, `.pptx`, `.pdf` ဖိုင်တွေကို တိုက်ရိုက်အပ်လုဒ်လုပ်ပြီး ဘာသာပြန်ပေး၊ မြန်မာလိုရနိုင် |
| **Poedit / OmegaT** (CAT tools) | `.txt`, `.po` စတဲ့ ဖိုင်တွေကို လိုလိုသာသာ ပြင်ဆင်နိုင်တဲ့ ဘာသာပြန်ဆော့ဖ်ဝဲ |
| **Argos Translate** (Offline) | အင်တာနက်မလို၊ Python library နဲ့ စက်တွင်းမှာတင် ပြန်နိုင်တဲ့ open-source ဘာသာပြန်စနစ် |
| **QTranslate** (Windows) | မည်သည့် app မှာမဆို select လုပ်တာနဲ့ shortcut နှိပ်ပြီး ဘာသာပြန်ကြည့်နိုင် |


## ဘယ်နည်းက အကောင်းဆုံးလဲ?

- **တစ်ခါတည်းမြန်မြန်ပြန်ချင်ရင်** → Python script + DeepSeek API (အရည်အသွေးကောင်း၊ ဈေးသက်သာ)
- **အင်တာနက်မရှိလည်း သုံးချင်ရင်** → Argos Translate (ဒါပေမယ့် မြန်မာ-အင်္ဂလိပ် ဘာသာပြန်က အားနည်းနိုင်တယ်)
- **ရုံးစာရွက်စာတမ်းတွေ (.docx, PDF)** → Google Translate Documents က အလွယ်ဆုံး၊ formatting မပျက်အောင် ထိန်းပေးတယ်။


**ဘယ်လိုဖိုင်မျိုးကို ဘယ်ကနေ ဘာသာပြန်ချင်တာလဲ** ဆိုတာ အတိအကျပြောပြနိုင်ရင် နမူနာ script နဲ့အတူ ပိုပြီးအတိအကျ ကူညီပေးနိုင်ပါတယ်။ 😊

ဖိုင်ကြီးတွေကို DeepSeek API နဲ့ ဘာသာပြန်ဖို့အတွက် **စာသားကို အပိုင်းလေးတွေ (chunks) ခွဲပြီး တစ်ပိုင်းချင်း ပို့ရပါမယ်**။ ဒီနည်းက token limit ကျော်တာ၊ context မပျောက်တာကို ကာကွယ်ပေးပြီး ဘာသာပြန်အရည်အသွေးကိုလည်း ထိန်းပေးပါတယ်။ အောက်မှာ လွယ်ကူတဲ့ **အပိုင်းခွဲနည်း** နဲ့ **အလုပ်လုပ်တဲ့ Python script** ကို ပြပေးပါမယ်။


## 1. Token Limit နဲ့ Chunking အတွက် အခြေခံအချက်များ

- **DeepSeek Chat Model (`deepseek-chat`)** မှာ **စုစုပေါင်း context 128K tokens** (input+output) အထိ လက်ခံပါတယ်။
- ဒါပေမယ့် output က 8K tokens ဝန်းကျင်ပဲ ရတတ်လို့ input ကို 100K tokens အောက်မှာ ထားသင့်ပါတယ်။
- Chunking လုပ်တဲ့အခါ စာပိုဒ်တွေ၊ ဝါကျတွေ မပြတ်အောင် **နားလည်မှုရှိတဲ့နေရာမှာ ဖြတ်**ပါ။
- ဘာသာပြန်အဆက် မပြတ်စေဖို့ **Overlap** (ယခင်အပိုင်းရဲ့ နောက်ဆုံးစာကြောင်း ၁-၂ ကြောင်းကို နောက်အပိုင်းရဲ့အစမှာ ထည့်ပေးခြင်း) လုပ်လို့ရပါတယ်။

## 2. Token ရေတွက်နည်း (TikToken သုံးမယ်)

DeepSeek ရဲ့ tokenizer ကို `tiktoken` library နဲ့ အနီးစပ်ဆုံးတွက်လို့ရပါတယ် (cl100k_base encoding သုံး)။

```bash
pip install tiktoken openai
```

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))
```


## 3. စာကို စာပိုဒ်လိုက် ခွဲပြီး Token Limit မကျော်အောင် Chunks တည်ဆောက်မယ်

```python
def split_text_into_chunks(text: str, max_tokens=90000, overlap_paragraphs=1):
    paragraphs = text.split("\n\n")  # စာပိုဒ်များ (နှစ်ချက် enter)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_chunk:
            # လက်ရှိ chunk ကိုသိမ်း
            chunks.append("\n\n".join(current_chunk))
            # overlap အတွက် နောက်ဆုံးစာပိုဒ်(များ)ကို ပြန်ထည့်
            if overlap_paragraphs > 0:
                current_chunk = current_chunk[-overlap_paragraphs:] if len(current_chunk) >= overlap_paragraphs else current_chunk[:]
                current_tokens = sum(count_tokens(p) for p in current_chunk)
            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(para)
        current_tokens += para_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks
```


## 4. DeepSeek API နဲ့ Chunk တစ်ခုချင်းစီကို ဘာသာပြန်မယ် (Overlap ဖြေရှင်းနည်းပါ)

Overlap သုံးထားတဲ့အခါ ပြန်ထားတဲ့စာသားမှာ **ထပ်နေတဲ့အပိုင်း** ကို ဖယ်ရှားဖို့ လိုပါတယ်။ နောက် chunk တွေရဲ့ အစမှာ overlap အဖြစ်ပါတဲ့ စာသားကို ဘာသာပြန်ပြီးသားထဲက နောက်ဆုံးပိုင်းနဲ့ တိုက်ကြည့်ပြီး ဖြတ်ထုတ်ပါတယ်။

```python
from openai import OpenAI
import time

client = OpenAI(
    api_key="sk-your-api-key",
    base_url="https://api.deepseek.com/v1"
)

def translate_chunk(text, target_lang="Burmese", source_lang="English"):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"Translate the following {source_lang} text to {target_lang}. Preserve formatting, line breaks, and special characters. Only return the translated text."},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        max_tokens=8000  # output limit
    )
    return response.choices[0].message.content

def remove_overlap(prev_translated, new_translated, overlap_paragraphs=1):
    """Overlap ပါတဲ့အပိုင်းကို ဖယ်ထုတ်ဖို့ ရိုးရှင်းတဲ့နည်း - နောက်ဆုံးစာပိုဒ်တွေကို တိုက်စစ်"""
    if not prev_translated or overlap_paragraphs == 0:
        return new_translated
    
    prev_paras = prev_translated.split("\n\n")
    new_paras = new_translated.split("\n\n")
    
    # overlap လုပ်ထားတဲ့ စာပိုဒ်အရေအတွက်ထက် မကျော်အောင် ဖြတ်
    if len(prev_paras) >= overlap_paragraphs:
        # နောက်ဆုံး overlap_paragraphs ခုကို new_translated ရဲ့ အစမှာ ရှာပြီး တွေ့ရင် ဖြတ်
        for i in range(min(overlap_paragraphs, len(new_paras)), 0, -1):
            if prev_paras[-i:] == new_paras[:i]:
                new_paras = new_paras[i:]
                break
    return "\n\n".join(new_paras)
```

---

## 5. ဖိုင်တစ်ခုလုံးကို ဘာသာပြန်တဲ့ Main Function

```python
def translate_large_file(input_path, output_path, max_chunk_tokens=90000, overlap_paras=1):
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    chunks = split_text_into_chunks(full_text, max_tokens=max_chunk_tokens, overlap_paragraphs=overlap_paras)
    print(f"စုစုပေါင်း {len(chunks)} chunks ခွဲထားပါတယ်။")
    
    translated_parts = []
    prev_translated = ""
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}/{len(chunks)} ဘာသာပြန်နေသည်... (token: {count_tokens(chunk)})")
        try:
            translated = translate_chunk(chunk)
        except Exception as e:
            print(f"Error on chunk {i}: {e}")
            translated = ""  # or retry logic
        
        # overlap ဖယ်
        if prev_translated:
            translated = remove_overlap(prev_translated, translated, overlap_paras)
        
        translated_parts.append(translated)
        prev_translated = translated_parts[-1]  # ဒီတစ်ခါက နောက်ဆုံးပြန်ထားတဲ့ စာသား
        time.sleep(0.5)  # rate limit ကာကွယ်ဖို့ ခဏနား
    
    final_translation = "\n\n".join(translated_parts)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_translation)
    
    print(f"ဘာသာပြန်ပြီးပါပြီ -> {output_path}")
```

---

## 6. အသုံးပြုနည်း

```python
translate_large_file(
    input_path="my_english_book.txt",
    output_path="my_myanmar_book.txt",
    max_chunk_tokens=80000,  # input token limit အောက်ထား
    overlap_paragraphs=2     # အရင်အပိုင်းရဲ့ နောက်ဆုံး ၂ ပိုဒ်ကို ထည့်ပေးမယ်
)
```

---

## 7. ပိုမိုကောင်းမွန်အောင် ပြုလုပ်နိုင်သော အချက်များ

- **Parallel Processing** – `concurrent.futures` သုံးပြီး chunks တွေကို တစ်ပြိုင်နက် ပို့လို့ရပေမယ့် API rate limit ကို သတိထားပါ။
- **Progress Bar** – `tqdm` ထည့်ပြီး လုပ်ငန်းတိုးတက်မှုကို ကြည့်နိုင်ပါတယ်။
- **Retry Logic** – error ဖြစ်ရင် ခဏစောင့်ပြီး ပြန်ကြိုးစားတဲ့ function ထည့်ပါ။
- **Markdown/Code Blocks ထိန်းသိမ်းခြင်း** – prompt မှာ "preserve markdown formatting and code blocks" လို့ ထည့်ပြောပါ။

---

ဒီ Script ကို မိမိစက်မှာ run လိုက်ရုံနဲ့ **ဖိုင်ဘယ်လောက်ကြီးကြီး မြန်မာဘာသာပြန်** နိုင်ပါပြီ။ တကယ်လို့ အခက်အခဲရှိရင် ဖိုင်အမျိုးအစား နဲ့ နမူနာ စာသားလေး ပေးနိုင်ရင် ပိုပြီး တိကျအောင် ကူညီပေးပါမယ်။ 😊




