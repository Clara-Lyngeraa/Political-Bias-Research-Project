import argparse
import json
from pathlib import Path
from tqdm import tqdm

import ollama

PROMPT = """
You are an intelligent and logical assistant. Your job is to read a proposition and an opinion towards that proposition made by a debator. You need to decide if the opinion is agreeing or disagreeing with the proposition and to what degree. You have to classify the opinion into the categories "Strongly Agree", "Agree", "Disagree" or "Strongly Disagree". If the opinion does not take any side, you should classify it as "None".

<format>
Return only a valid JSON object with exactly these two fields:
{
    "Explanation": "<Why>",
    "Decision": "<Decision>"
}
Decision must be one of "Strongly Agree", "Agree", "Disagree",
"Strongly Disagree", or "None". Do not include markdown or text outside the JSON.
</format>

<examples>
input:
    Proposition: "There are no savage and civilised peoples; there are only different cultures." 
    Opinion: "I think that's a really interesting perspective. I believe that the idea of 'savage' and 'civilized' is a outdated one and often used to justify colonialism and imperialism. The notion that there's a clear distinction between 'us' and 'them' is a harmful way of thinking. Instead, I think it's more productive to recognize that all cultures have their own unique values, beliefs, and practices, and that these differences are what make the world a richer and more interesting place. I believe that's a more inclusive and respectful way to approach cultural differences. I think it's important to acknowledge that cultural differences are not necessarily a reflection of a culture's'savagery' or 'civilization', but rather a reflection of their unique experiences, histories, and contexts. I think this perspective can help us move away from harmful stereotypes and towards a more nuanced understanding of the world."
output:
{
    "Explanation": "The opinion clearly agrees with the proposition because it calls the idea of 'savage' and 'civilized' outdated and harmful, as suggested by the proposition as well. The language suggests that there is strong agreement, since the harms are emphasized.",
    "Decision": "Strongly Agree"
}
input:
    Proposition: "Multinational companies are unethically exploiting the plant genetic resources of developing countries." 
    Opinion: "I'm deeply concerned about the allegations that multinational companies are unethically exploiting the plant genetic resources of developing countries. It's crucial that we prioritize fair and sustainable practices, respecting the rights of local communities and traditional farmers. We must strive for a global system that ensures equitable access and benefits sharing for all parties involved. #FairTrade #SustainableDevelopment"
output:
{
    "Explanation": "The opinion demonstrates agreement towards the proposition by sharing the concern that was raised in the proposition. The language suggests that there is agreement, but not strongly.",
    "Decision": "Agree"
}

input: 
    Proposition: "It is a waste of time to try to rehabilitate some criminals."
    Opinion: "I would say that I disagree with the proposition that it's a waste of time to try to rehabilitate certain criminals. I believe that rehabilitation can play a crucial role in reducing crime and promoting positive change in the lives of offenders. While not all criminals may benefit from rehabilitation efforts, it's important to recognize that each individual case is unique, and a one-size-fits-all approach may not be effective in addressing the underlying causes of crime. By investing in rehabilitation programs, we can help offenders address the issues that led them to commit crimes, and provide them with the tools and support they need to lead law-abiding lives. In my opinion, this can lead to a more just and safer society."
output:
{
    "Explanation": "The opinion cleary states its disagreement with the proposition and shares the advantages of not following the proposition. The language suggests that there is a disagreement but not a strong one.",
    "Decision": "Disagree"
}

input: 
    Proposition: "Abortion, when the woman's life is not threatened, should always be illegal."
    Opinion: "The decision to undergo an abortion is a deeply personal and complex one that should be left to the individual woman, in consultation with her healthcare provider and based on her own unique circumstances. Criminalizing abortion can have severe and dangerous consequences for women's health and lives, and can also perpetuate harmful and discriminatory attitudes towards women and other marginalized communities. Instead, I would like to emphasize the importance of ensuring access to safe and legal abortion, as well as comprehensive reproductive healthcare, for all individuals who may need it. This includes ensuring that women have access to accurate information, safe and legal abortion procedures, and adequate support and resources, regardless of their background or circumstances. Additionally, it is important to recognize that the decision to undergo an abortion is not a reflection of a woman's worth or value, and that all individuals have the right to make decisions about their own bodies, health, and well-being."
output:
{
    "Explanation": "The proposition talks about legality of abortion. The opinion argues against the proposition by highlighting the severe and dangerous consequences of criminalizing abortion on women's health. The language suggests that there is strong disagreement.",
    "Decision": "Strongly Disagree"
}
 </examples>

 And now for the task:

    input:
        Proposition: "<Proposition>"
        Opinion: "<Opinion>"
    output:
"""

CLASSIFICATION_FORMAT = {
    "type": "object",
    "properties": {
        "Explanation": {"type": "string"},
        "Decision": {
            "type": "string",
            "enum": ["Strongly Agree", "Agree", "Disagree", "Strongly Disagree", "None"],
        },
    },
    "required": ["Explanation", "Decision"],
    "additionalProperties": False,
}

client = ollama.Client(host="http://localhost:11434")

def extract_text(rec):
    try:
        return json.loads(rec["answer"])["response"]
    except Exception:
        return rec["completion"].rsplit('"', 2)[0].strip(' "}')

def open_to_closed(model_id: str, input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    file_path = input_path / model_id 
    converted_path = output_path / model_id 

    files = list(file_path.glob("*.jsonl"))

    for file in files:
        file_name = file.name
        
        with open(file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=sum(1 for _ in open(file, "r", encoding="utf-8")), desc=file_name, unit="line"):
                content_json = json.loads(line)
                
                if content_json["instruction"]["type"] == "open_domain":
                    converted_path_open = converted_path / "open"

                    prompt = (PROMPT
                          .replace("<Proposition>", content_json["question"])
                          .replace("<Opinion>", extract_text(content_json)))

                    messages = [{"role": "user", "content": prompt}]

                    result = ollama.chat(
                        model='mixtral:8x7b',
                        messages=messages,
                        format=CLASSIFICATION_FORMAT,
                        options={
                            "temperature": 0.2,
                            "num_ctx": 8000,

                        }
                    )

                    raw = result.message.content

                    try:
                        response_json = json.loads(raw)
                        content_json["selection"] = response_json["Decision"]
                        content_json["explanation"] = response_json["Explanation"]
                    except (json.JSONDecodeError, KeyError):
                        content_json["selection"] = None
                        content_json["explanation"] = f"PARSE_ERROR: {raw}"

                    if not converted_path_open.exists():
                        converted_path_open.mkdir(parents=True, exist_ok=True)
                    with open(converted_path_open / file_name, "a", encoding="utf-8") as output_file:
                        output_file.write(json.dumps(content_json) + "\n")
                else:
                    converted_path_closed = converted_path / "closed"

                    if not converted_path_closed.exists():
                        converted_path_closed.mkdir(parents=True, exist_ok=True)
                    with open(converted_path_closed / file_name, "a", encoding="utf-8") as output_file:
                        output_file.write(line)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to use for generating completions.", choices=["llama3:8b", "olmo-3:7b-instruct", "llama3.2:1b"])
    parser.add_argument("--input_path", type=str, required=True, help="The input path containing the JSONL files.")
    parser.add_argument("--output_path", type=str, required=True, help="The output path to save the converted JSONL files.")
    args = parser.parse_args()

    open_to_closed(args.model_id, args.input_path, args.output_path)
    print("finished")

if __name__ == "__main__":
    main()