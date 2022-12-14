import aiogram #asyncio
import uuid
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types import InputFile, ChatActions, InlineKeyboardMarkup, InlineKeyboardButton
import os
from dotenv import load_dotenv
from definitions import INPUT_IMAGES_DIR, IMAGES_DIR, ROOT_DIR, OUTPUT_IMAGES_DIR
from img.image_styling import process_im

load_dotenv()

BOT_KEY = os.getenv('API_KEY')

bot = Bot(token=BOT_KEY)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply('Hello! How are you?')


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
    await message.reply(message.text)


executor.start_polling(dp)