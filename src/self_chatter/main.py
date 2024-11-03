import os
import random
import tomllib
import typing as t

import click
import duckdb
import sqlmodel
import tomli_w
from bs4 import BeautifulSoup
from langchain_community.vectorstores import DuckDB
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from mastodon import Mastodon

config_home_path = os.path.expanduser("~/.config/self_chatter")
os.makedirs(config_home_path, exist_ok=True)
config_path = os.path.join(config_home_path, "config.toml")
vector_db_path = os.path.join(config_home_path, "vector.db")
data_db_path = os.path.join(config_home_path, "data.db")


class Text(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    content: str
    handle: str


sqlite_url = f"sqlite:///{data_db_path}"
engine = sqlmodel.create_engine(sqlite_url, echo=False)
sqlmodel.SQLModel.metadata.create_all(engine)

if os.path.exists(config_path) is False:
    default_config = {
        "handles": [],
        "embedding_model": "llama3.2:1b",
        "chat_model": "llama3.2",
    }
    with open(config_path, "w") as f:
        f.write(tomli_w.dumps(default_config))
with open(config_path, "r") as f:
    config = tomllib.loads(f.read())


def parse_handle(handle: str) -> t.Tuple[str, str]:
    parts = handle.split("@")
    username = parts[1]
    api_base = f"https://{parts[2]}"
    return username, api_base


def get_vector_store():
    duckdb_conn = duckdb.connect(
        database=vector_db_path,
        config={
            "enable_external_access": "false",
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
        },
    )
    embeddingfn = OllamaEmbeddings(model=config.get("embedding_model", "llama3.2:1b"))
    return DuckDB(connection=duckdb_conn, embedding=embeddingfn)


@click.group(help="Self chatter! Please have Ollama running.")
def cli():
    pass


@cli.command(help="Pull data from your social network posts.")
@click.option(
    "--max-toots-per-handle",
    default=1000,
    help="number of toots to pull per handle (default: 1000)",
)
@click.option(
    "--handles",
    default="",
    help='comma separated handle list (for example, "@me@mysite.com, @me@frensite.com"), this overwrites handles in the config file (default: "")',
)
def pulldata(max_toots_per_handle, handles):
    vector_store = get_vector_store()
    click.echo("Pulling data from online...")
    if handles:
        handles_from_args = [handle.strip() for handle in handles.split(",")]
    else:
        handles_from_args = None
    for handle in handles_from_args or config.get("handles", []):
        cnt = 0
        click.echo(f"Working on {handle}")
        username, api_base = parse_handle(handle)
        mastodon_client = Mastodon(api_base_url=api_base)
        page_size = 40
        max_id = None
        while True:
            with sqlmodel.Session(engine) as session:
                data = mastodon_client.account_statuses(
                    username, limit=page_size, max_id=max_id
                )
                cnt += len(data)
                max_id = data[-1]["id"]
                for toot in data:
                    text = toot["content"] or ""
                    text = text.strip()
                    soup = BeautifulSoup(text, "lxml")
                    text = soup.get_text().strip()
                    if not text:
                        continue
                    statement = (
                        sqlmodel.select(Text)
                        .where(Text.content == text)
                        .where(Text.handle == handle)
                    )
                    existing_text = session.exec(statement).first()
                    if not existing_text:
                        vector_store.add_texts(CharacterTextSplitter().split_text(text))
                        session.add(Text(content=text, handle=handle))
                        session.commit()
                if len(data) < page_size:
                    break
                if cnt >= max_toots_per_handle:
                    break
    click.echo("Finished pulling data from online.")


@cli.command(help="Start chatting with existing data")
@click.option("--verbose", default=False, help="verbose output (default: False)")
def chat(verbose):
    vector_store = get_vector_store()
    llm = ChatOllama(model=config.get("chat_model", "llama3.2"))
    initial_system_message = "You are an online persona who's popular on decentralzed social media talking to your IRL self, i.e., the user."
    results = vector_store.similarity_search(initial_system_message, k=10)
    system_message = initial_system_message + "Here are some relevant posts:"
    for result in results:
        system_message += result.page_content + ";"
    messages = [
        ("system", system_message),
        ("human", "Now greet the user."),
    ]
    if verbose:
        click.echo(messages)

    resp = llm.invoke(messages)
    messages.append(
        (
            resp.response_metadata["message"]["role"],
            resp.content,
        )
    )
    click.echo(f"AI You: {resp.content}")
    while True:
        try:
            raw_human_message = input("You: ")
            results = vector_store.similarity_search(raw_human_message, k=10)
            human_message = "Here are some posts related to your conversation:"
            for result in results:
                human_message += result.page_content + ";"
            human_message += f"\n\n Be precise and to the point, do not say too much. Here is the user's message that you need to reply to: {raw_human_message}"
            messages.append(("human", human_message))
            resp = llm.invoke(messages)
            messages.append(
                (
                    resp.response_metadata["message"]["role"],
                    resp.content,
                )
            )
            if verbose:
                click.echo(messages)
            click.echo(f"AI You: {resp.content}")
        except KeyboardInterrupt:
            break


@cli.command(help="Create a post using the online persona's tone.")
def post():
    with sqlmodel.Session(engine) as session:
        count = session.exec(
            sqlmodel.select(sqlmodel.func.count(sqlmodel.col(Text.id)))
        ).one()
        if count == 0:
            click.echo("You haven't pulled any posts from the internet!")
            return
        sample_size = 15
        if count <= sample_size:
            previous_posts = session.exec(sqlmodel.select(Text)).all()
        else:
            previous_post_ids = random.sample(range(1, count + 1), sample_size)
            previous_posts = [
                session.exec(sqlmodel.select(Text).where(Text.id == post_id)).one()
                for post_id in previous_post_ids
            ]

    system_message = "You are a famous social media user. Here are some of your posts: "

    for previous_post in previous_posts:
        system_message = f"{system_message} \n\n {previous_post.content}"
    messages = [
        ("system", system_message),
        ("human", "Create a new post in the same tone and style as the old posts."),
    ]
    llm = ChatOllama(model=config.get("chat_model", "llama3.2"))
    resp = llm.invoke(messages)
    click.echo(resp.content)


if __name__ == "__main__":
    cli()
