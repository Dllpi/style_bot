import aiogram #asyncio
import uuid
import logging
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types import InputFile, ChatActions, InlineKeyboardMarkup, InlineKeyboardButton
import os
from dotenv import load_dotenv
from definitions import INPUT_IMAGES_DIR, IMAGES_DIR, ROOT_DIR, OUTPUT_IMAGES_DIR
from img.image_styling import process_im
import markup as nav
from db import DataBase


load_dotenv()

logging.basicConfig(level=logging.INFO)
BOT_KEY = os.getenv('API_KEY')

bot = Bot(token=BOT_KEY)
dp = Dispatcher(bot)
db = DataBase('database')


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    if not db.user_exists(message.from_user.id):
        db.add_user(message.from_user.id)
        await bot.send_message(message.from_user.id, 'Введи свой ник: ')
    else:
        await bot.send_message(message.from_user.id, 'Ты уже зарегистрирован .', reply_markup=nav.mainMenu)

'''
@dp.message_handler()
async def bot_message(message: types.Message):

    if message.text == 'Profile':
        user_nickname = 'Ur nickname: ' + db.get_nickname(message.from_user.id)
        await bot.send_message(message.from_user.id, user_nickname)

    else:
        if db.get_signup(message.from_user.id) == 'setnickname':
            if len(message.text) > 15:
                await bot.send_message(message.from_user.id, 'Ur nick is so long')
            else:
                db.set_nikname(message.from_user.id, message.text)
                db.get_signup(message.from_user.id)
                await bot.send_message(message.from_user.id, 'Hi ur in side me :) Just send photo for me ', reply_markup=nav.mainMenu)
        else:
            await message.reply(message.text)
'''

@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply('whats wrong?')


@dp.message_handler(content_types=['photo'])
async def process_photo(message: types.Message):
    filename = uuid.uuid4().hex[:12].upper()
    print(filename)
    await message.photo[-1].download(os.path.join(INPUT_IMAGES_DIR, f"{filename}.jpg"))
    await bot.send_chat_action(message.from_user.id, ChatActions.TYPING)
    await bot.send_message(message.from_user.id, "Ваше фото успешно получено :)")

    rkm = InlineKeyboardMarkup(row_width=3)
    artists = [('Ван Гог', 'model_van-gogh'), ('Клод Моне', 'model_monet'), ('Василий Кандинский', 'model_kandinsky')]

    for artist in artists:
        print(artist)
        rkm.add(InlineKeyboardButton(artist[0], callback_data=f'{artist[1]}@{filename}.jpg'))

    await message.reply('Выберите стиль:', reply_markup=rkm)


@dp.callback_query_handler(func=lambda x: x.data.startswith('model'))
async def choose_style(style: types.CallbackQuery):
    model_name, file_name = style.data.split('@')
    res = process_im(file_name=file_name, model_name=model_name)
    if res:
        photo = InputFile(os.path.join(OUTPUT_IMAGES_DIR, f"{file_name.split('.')[0]}_stylized.jpg"))
        await style.message.reply_photo(photo=photo, caption="Вот ваше фото:")
    await style.message.reply(text="Вы можете отправить ещё одно фото...")


@dp.message_handler()
async def process_message(message: types.Message):
    if message.chat.type == 'private':
        if message.text == 'Profile':
            user_nickname = 'Твой ник: ' + db.get_nickname(message.from_user.id)
            await bot.send_message(message.from_user.id, user_nickname)

        else:
            if db.get_signup(message.from_user.id) == 'setnickname':
                if len(message.text) > 15:
                    await bot.send_message(message.from_user.id, 'Твой ник, слишком длинный')
                else:
                    db.set_nikname(message.from_user.id, message.text)
                    db.get_signup(message.from_user.id)
                    await bot.send_message(message.from_user.id, 'Привет, ты зарегестрировался , отправь мне фото. \n Если ты отправишь мне не фотографию....\n Я поменяю твой ник , будет тебе уроком',
                                           reply_markup=nav.mainMenu)
            else:
                await message.reply(message.text)

executor.start_polling(dp)
