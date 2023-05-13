from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

btnProfile = KeyboardButton('Profile')


mainMenu = ReplyKeyboardMarkup(resize_keyboard=True)
mainMenu.add(btnProfile)
