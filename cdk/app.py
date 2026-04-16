import aws_cdk as cdk
from lib.rag_stack import RAGStack

app = cdk.App()
RAGStack(app, "RAGStack")
app.synth()
