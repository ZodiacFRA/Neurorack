"""

 ~ Neurorack project ~
 Menu : Set of classes for handling the menus
 
 This file defines the main operations for graphical menus
 The functions here will be used for the LCD display.
 Parts of this code have been inspired by the great piControllerMenu code:
     https://github.com/Avanade/piControllerMenu
 
 Author               :  Ninon Devis, Philippe Esling, Martin Vert
                        <{devis, esling}@ircam.fr>
 
 All authors contributed equally to the project and are listed aphabetically.

"""
import yaml
from .graphics import ScrollableGraphicScene
from .config import config
from .menu_items import MenuItem

class Menu(ScrollableGraphicScene):
    '''
        Main class for the LCD Menu. 
        Handles loading and running the menu and commands. 
    '''
    
    def __init__(self, 
                 config_file = "./menu.yaml",
                 x:int = 10,
                 y:int = 10,
                 height = 240,
                 width = 180,
                 absolute = True,
                 screen_signal = None):
        '''
            Constructor. Creates a new instance of the ContollerMenu class. 
            Paramters: 
                config:     str
                            Name of the config file for the controller menu. Defaults to ./controllerMenu.yaml. 
        '''
        super().__init__(x, y, absolute, True, height, width)
        self._config_file = config_file
        self._config = None
        self._root_menu = None
        self._items = {}
        self._current_items = []
        self._history = [""]
        self._mode = config.menu.mode_basic
        self._screen_signal = screen_signal
        self.load()

    def load(self):
        '''
            Loads the controller menu configuration and boots the menu. 
        '''
        # Load the menu
        with open(self._config_file) as file:
            # The FullLoader handles conversion from scalar values to Python dict
            self._config = yaml.load(file, Loader=yaml.FullLoader)
        self._root_menu = self._config["root"]
        self._current_menu = self._root_menu
        # Generate all items from menu
        for item in self._config["items"]:
            self._items[item] = MenuItem.create_item(item, self._config["items"][item])
        print(self._items)
        print(self._current_menu)
        self.generate_current_elements()
    
    def generate_current_elements(self):
        self._elements = []
        for item in self._current_menu:
            if (type(self._current_menu[item]) == dict):
                self._elements.append(MenuItem(title = item, type = 'menu', command = ''))
            else:
                self._elements.append(self._items[self._current_menu[item]])
        if (self._current_menu == self._root_menu):
            self._elements.append(MenuItem(title = config.menu.exit_element, type = 'menu', command = ''))
        else:
            self._elements.append(MenuItem(title = config.menu.back_element, type = 'menu', command = ''))

    def process_select(self, 
                       select_index: int, 
                       select_item: str):
        """
            Delegate to respond to a select event on the controller tactile select button. Invokes either
            navigation to a submenu or command execution. 
            Paramters: 
                select_index:   [int]
                                Index of the menu item selected.
                select_item:    [str]
                                The selected menu item
        """
        items = [".."]
        if self._elements[select_index]._type != 'menu':
            print(f"Load {self._elements[select_index]._title}")
            self._current_menu = self._current_menu[select_item._title]
            self._history.append(select_item._title)
            self.generate_current_elements()
            self.reset_menu()
        else:
            print(f"Execute {self._elements[select_index]._title}")
            self._items[self._current_menu[select_item]].run()

    def process_history(self):
        """
            Delegate to respond to the navigate uo event on the controller tactile Up button. Loads the 
            previous menu. 
        """
        items = []
        self._history.pop()
        for level in self._history:
            if level == "": 
                self._current_menu = self._root_menu
            else: 
                self._current_menu = self._current_menu[level]
        self.generate_current_elements()

    def process_confirm(self, command:any, confirmState: int):
        """
            Delegate to respond to the confirmation event from the confirmation screen. Depending on event state, 
            either reloads the previous menu (confirmState==CONFIRM_CAMCEL) or run the commmand (CONFIRM_OK)
            Parameters:
                command:        Command
                                The command to run if confirmed
                confirmState:   int
                                The confirm state. Either CONFIRM_OK or CONFIRM_CANCEL
        """
        if confirmState == config.menu.confirm_cancel: self.__disp.DrawMenu()
        else:
            command.Run(display=self.__disp, confirmed=config.menu.confirm_ok)    
        
    def Spinner(self, run=True):
        """
            Loads the spinner screen while a command executes. This will start a background thread when invoked with run=True and 
            derminate the thread (and spinner) when invoked with run=False
            Parameters:
                run:    bool
                        Optional. Pass True to start the spinner, false to derminate the spinner.
        """
        if run==True:
            self.__stopSpinner = False
            self.__spinnerThread = threading.Thread(target=self.__drawSpinner, name="spinner", args=(lambda : self.__stopSpinner, ))
            self.__spinnerThread.start()
        else:
            if self.__spinnerThread is None: return
            self.__stopSpinner = True
            self.__spinnerThread.join()

    def navigation_callback(self, state, event_type):
        """
            Delegate called by the Navigation model when a navigation event occurs on the GPIO. Handles 
            corresponding invokation of the various display draw and/or command execution delegates. 
            Parameters:
                eventType:  int
                            The type of event that occured. One of the following:
                                DOWN_CLICK
                                UP_CLICK
                                LEFT_CLICK
                                RIGHT_CLICK
                                SELECT_CLICK
        """
        if (event_type == 'rotary'):
            direction = state['rotary_delta'].value
        if (self._mode == config.menu.mode_basic):
            if (event_type == 'rotary' and direction > 0):
                if self._selected_index == self._max_index - 1 and self._scroll_down is False: 
                    return
                if (self._selected_index >= 0):
                    self._elements[self._selected_index]._graphic._selected = False
                if self._selected_index == self._max_index - 1: 
                    self._scroll_start +=1
                self._selected_index += 1
                self._elements[self._selected_index]._graphic._selected = True
                return
            if (event_type == 'rotary' and direction < 0):
                if self._selected_index == 0 and self._scroll_up is False: 
                    return
                if self._selected_index == -1: 
                    self._selected_index = 0
                    self._scroll_start = 0
                else:
                    self._elements[self._selected_index]._graphic._selected = False
                    if self._selected_index == self._scroll_start: 
                        self._scroll_start -= 1
                    self._selected_index -= 1
                    self._elements[self._selected_index]._graphic._selected = True
                return
            if (event_type == 'button'):
                if self._elements[self._selected_index]._title == config.menu.back_element:
                    self.process_history()
                    self.reset_menu()
                elif self._elements[self._selected_index]._title == config.menu.exit_element:
                    self.reset_menu()
                    state["screen"]["mode"].value = config.screen.mode_main
                    self._screen_signal.set()
                    return
                elif self._selected_index > -1: 
                    self.process_select(self._selected_index, self._elements[self._selected_index])
                return
        
        """
        if eventType is LEFT_CLICK and self.__mode == MODE_MENU and self.__upCallback: 
            self.__upCallback()
            return
        if eventType is SELECT_CLICK and self.__mode == MODE_CONFIRM and self.__confirmCallback:
            self.__confirmCallback(self.__confirmCommand, self.__confirmState)
            return
        if eventType is SELECT_CLICK and self.__mode == MODE_OUTPUT:
            self.DrawMenu()
            return
        if self.__mode == MODE_EXTERNAL and (eventType is SELECT_CLICK or eventType is LEFT_CLICK):
            self.__stopCommand = True
            return
        if eventType is LEFT_CLICK and self.__mode == MODE_CONFIRM: 
            self.DrawConfirmation(state=CONFIRM_OK)
            return
        if eventType is RIGHT_CLICK and self.__mode == MODE_CONFIRM: 
            self.DrawConfirmation(state=CONFIRM_CANCEL)
            return
        """

    def reset_menu(self):
        """
            Resets the current menu to an unselected state.
        """
        self._selected_index = -1
        #self.DrawMenu()

class MenuBar():
    def __init__(self):
        pass

if __name__ == '__main__':
    menu = Menu('../menu.yaml')
    