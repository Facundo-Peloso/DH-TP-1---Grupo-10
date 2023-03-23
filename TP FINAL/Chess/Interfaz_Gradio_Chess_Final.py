import pickle
import pandas as pd
import gradio as gr
import sklearn 


# Cargar el modelo
with open('modelo_search_2.pkl', 'rb') as f:
    model = pickle.load(f)
    
def predict(mainline_moves, Opening, WhiteFideId, BlackFideId, WhiteElo, BlackElo):
    df_model = pd.DataFrame({"mainline_moves":mainline_moves,
                             "Opening":Opening,
                             "WhiteFideId":WhiteFideId,
                             "BlackFideId":BlackFideId,
                             "WhiteElo": WhiteElo, 
                             "BlackElo": BlackElo}, 
                            index= [0])
    
    label_pred = model.predict(df_model)[0].item()
    
    if label_pred == 0:
        return "Ganó Blancas"
    elif label_pred == 1:
        return "Ganó Negras"
    else:
        return "Tablas"
    
# Definir los widgets de entrada

mainline_moves = gr.inputs.Number(label="Mainline Moves")
Opening = gr.inputs.Dropdown(["sicilian", "qgd", "english", "reti",
                              "french", "king's indian", "queen's pawn game",
                              "ruy lopez", "caro-kann", "nimzo-indian",
                              "queen's indian", "giuoco piano", "gruenfeld",
                              "catalan"],
                             label="Opening",
                             )                             
WhiteFideId = gr.inputs.Dropdown([1503014, 4168119, 13400924,
                                  5000017, 5202213, 2020009,
                                  2016192, 24116068, 12573981,
                                  8603677, 13300474, 738590, 4126025,
                                  3503240, 13401319, 623539,
                                  14204118,46616543, 5029465, 4158814 ],
                                 label="White Fide ID")
BlackFideId = gr.inputs.Dropdown([1503014, 4168119, 13400924,
                                  5000017, 5202213, 2020009,
                                  2016192, 24116068, 12573981,
                                  8603677, 13300474, 738590, 4126025,
                                  3503240, 13401319, 623539,
                                  14204118,46616543, 5029465, 4158814 ],
                                label="Black Fide ID")
WhiteElo = gr.inputs.Slider(label="White Elo", 
                            minimum=0, maximum=3000, step=10,
                            default=2400)
BlackElo = gr.inputs.Slider(label="Black Elo", 
                            minimum=0, maximum=3000, step=10,
                            default=2400)

# Definir la salida
output_label = gr.outputs.Label(num_top_classes=1, label="Resultado de la partida")

# Definir la interfaz
iface = gr.Interface(fn=predict, 
                     inputs=[mainline_moves, Opening, WhiteFideId, BlackFideId, WhiteElo, BlackElo], 
                     outputs=output_label, 
                     title="Predicción de partidas de Ajedrez", 
                     theme=gr.themes.Soft(),
                     layout="horizontal",
                     allow_flagging=False,
                     description="Coloca los datos solicitados y descubre el resultado de la partida!")

# Ejecutar la interfaz
iface.launch()
