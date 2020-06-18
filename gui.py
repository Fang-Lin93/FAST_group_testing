from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.config import Config
import sim

Config.set('graphics', 'width', '700')
Config.set('graphics', 'height', '700')
Config.set('graphics', 'resizable', '0')

SaveInput = ""


class GroupTestApp(App):
    def __init__(self, **kwargs):
        super(GroupTestApp, self).__init__(**kwargs)
        self.result = TextInput(readonly=True, font_size=32, size_hint=[1, .75], background_color=[1, 1, 1, .8])
        self.household = TextInput(hint_text="Household distribution", text="([1, 3, 5, 7], [0.02, 0.3, 0.46, 0.22])",
                                   font_size=32, size_hint=[1, .75], multiline=False)
        self.rho = TextInput(hint_text="Prevalence", text="0.005", font_size=32, size_hint=[1, .75],
                             input_filter='float', multiline=False)
        self.beta = TextInput(hint_text="Infection rate", text="0.7", font_size=32, size_hint=[1, .75],
                              input_filter='float', multiline=False)
        self.size = TextInput(hint_text="Size", font_size=32, text="20000", size_hint=[1, .75],
                              input_filter='float', multiline=False)
        self.replication = TextInput(hint_text="Replication", text="50", font_size=32, size_hint=[1, .75],
                                     input_filter='float', multiline=False)
        self.fnr = TextInput(hint_text="FNR", text="0.02", font_size=32, size_hint=[1, .75],
                             input_filter='float', multiline=False)
        self.fpr = TextInput(hint_text="FPR", text="0.001", font_size=32, size_hint=[1, .75],
                             input_filter='float', multiline=False)

    def build(self):
        root = BoxLayout(orientation='vertical', padding=1)
        root.add_widget(self.result)
        paras = GridLayout(cols=3)
        paras_list = [self.rho, self.beta, self.size,
                      self.replication, self.fnr, self.fpr]
        for i in paras_list:
            paras.add_widget(i)
        root.add_widget(paras)
        root.add_widget(self.household)

        methods = GridLayout(cols=4)
        methods.add_widget(Button(text='Individual', on_press=self.calculate))
        methods.add_widget(Button(text='Dorfman', on_press=self.calculate))
        methods.add_widget(Button(text='FeatCostly', on_press=self.calculate))
        methods.add_widget(Button(text='FeatCheap', on_press=self.calculate))
        root.add_widget(methods)

        return root

    def calculate(self, click):
        if click:

            try:
                paras_sample = {
                    'size': eval(self.size.text),
                    'rho': eval(self.rho.text),
                    'household': eval(self.household.text),
                    'beta': eval(self.beta.text),
                    'replication': eval(self.replication.text)
                }

                test = {
                    'fnr': eval(self.fnr.text),
                    'fpr': eval(self.fpr.text),
                    'dilution_factor': 0.0
                }
                running = sim.Optimal(**paras_sample)
                res, opt = running.get_opt(click.text, **test)

                display = 'Results:' + '\n'
                for key, i in res.items():
                    display += key + '=' + str(i) + '\n'
                display += 'Optimal:' + '\n'
                for key, i in opt.items():
                    display += key + '=' + str(i) + '\n'
                display += 'm=' + str(running.m)
                self.result.text = display

            except Exception:
                self.result.text = 'Error'


if __name__ == '__main__':
    GroupTestApp().run()
