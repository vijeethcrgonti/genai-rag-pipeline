import aws_cdk as cdk
from lib.rag_stack import RagStack

app = cdk.App()
RagStack(app, "RagStack")
app.synth()
