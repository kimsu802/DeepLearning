import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from PIL import Image
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import io
import plotly.express as px

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)


net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


PATH = "model.pth"  # 모델의 경로
net.load_state_dict(torch.load(PATH))
net.eval()


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


app = dash.Dash(__name__)


app.layout = html.Div([

    html.H1(children='이미지 업로드를 통한 이미지 분류'),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            '드래그 앤 드랍하거나 ',
            html.A('이미지 파일을 업로드 해주세요.')
        ]),
        style={
            'width': '50%',
            'height': '50px',
            'lineHeight': '50px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='output-accuracy')
])


@app.callback(
    [Output('output-image-upload', 'children'), Output('output-accuracy', 'children')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        # base64 인코딩된 이미지 데이터를 디코딩하여 PIL 이미지로 변환
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

      
        processed_image = preprocess_image(image)
        processed_image = processed_image.to(device)

      
        outputs = net(processed_image)
        _, predicted = torch.max(outputs, 1)
        prediction = classes[predicted.item()]

       
        result_div = html.Div([
            html.H3('업로드된 이미지:'),
            html.Img(src=contents, style={'height': '300px'}),
            html.H3('예측 이미지는: {}'.format(prediction))
        ])

       
        probabilities = F.softmax(outputs, dim=1)[0].cpu().detach().numpy()
        accuracy_div = html.Div([
            html.H3('각 클래스 별 정확도 :')
        ])
        for i, class_name in enumerate(classes):
            accuracy = probabilities[i]
            accuracy_text = f'{class_name}: {accuracy:.4f}'
            accuracy_div.children.append(html.P(accuracy_text))

       
        data = {
            'Class': classes,
            'Accuracy': probabilities
        }
        fig = px.bar(data, x='Class', y='Accuracy')
        graph_div = dcc.Graph(figure=fig)

        return result_div, [accuracy_div, graph_div]
    else:
        return None, None

if __name__ == '__main__':
    app.run_server(debug=True)
