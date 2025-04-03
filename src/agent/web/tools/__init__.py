from src.agent.web.tools.views import Click,Type,Wait,Scroll,GoTo,Back,Key,Download,ExtractContent,Tab,Upload,Menu,Form
from main_content_extractor import MainContentExtractor
from src.agent.web.context import Context
from typing import Literal
from src.tool import Tool
from pathlib import Path
from os import getcwd
import httpx

@Tool('Click Tool',params=Click)
async def click_tool(index:int,context:Context=None):
    '''For clicking buttons, links, checkboxes, and radio buttons'''
    page=await context.get_current_page()
    element,handle=await context.get_element_by_index(index)
    if element.attributes.get('type','') in ['checkbox','radio']:
        await page.wait_for_load_state('load')
        await handle.check(force=True)
        return f'Checked element at index {index}'
    else:
        await page.wait_for_load_state('load')
        await handle.scroll_into_view_if_needed()
        await handle.click()
        return f'Clicked element at index {index}'


@Tool('Type Tool',params=Type)
async def type_tool(index:int,text:str,context:Context=None):
    '''To fill input fields or search boxes'''
    page=await context.get_current_page()
    _,handle=await context.get_element_by_index(index)
    await page.wait_for_load_state('load')
    await handle.scroll_into_view_if_needed()
    await handle.type(text,delay=80)
    return f'Typed {text} in element {index}'

@Tool('Wait Tool',params=Wait)
async def wait_tool(time:int,context:Context=None):
    '''To wait until the page has fully loaded before proceeding'''
    page=await context.get_current_page()
    await page.wait_for_timeout(time*1000)
    return f'Waited for {time}s'

@Tool('Scroll Tool',params=Scroll)
async def scroll_tool(direction:str,amount:int=None,context:Context=None):
    '''To scroll the page by a certain amount or by a page'''
    page=await context.get_current_page()
    if direction=='up':
        if amount is None:
            await page.keyboard.press('PageUp')
        else:
            await page.mouse.wheel(0,-amount)
    elif direction=='down':
        if amount is None:
            await page.keyboard.press('PageDown')
        else:
            await page.mouse.wheel(0,amount)
    else:
        raise ValueError('Invalid direction')
    amount=amount if amount else 'one page'
    return f'Scrolled {direction} by {amount}'

@Tool('GoTo Tool',params=GoTo)
async def goto_tool(url:str,context:Context=None):
    '''To navigate directly to a specified URL.'''
    page=await context.get_current_page()
    await page.goto(url)
    return f'Navigated to {url}'

@Tool('Back Tool',params=Back)
async def back_tool(context:Context=None):
    '''To return to the previous page'''
    page=await context.get_current_page()
    await page.go_back()
    await page.wait_for_load_state('load')
    return 'Navigated to previous page'

@Tool('Key Tool',params=Key)
async def key_tool(keys:str,context:Context=None):
    '''To perform keyboard shorcuts'''
    page=await context.get_current_page()
    await page.keyboard.press(keys)
    return f'Pressed {keys}'

@Tool('Download Tool',params=Download)
async def download_tool(index:int=None,url:str=None,filename:str=None,context:Context=None):
    '''To download a file (e.g., pdf, image, video, audio) to the system'''
    folder_path=Path(getcwd()).joinpath('./downloads')
    folder_path.mkdir(parents=True,exist_ok=True)
    try:
        page=await context.get_current_page()
        _,handle=await context.get_element_by_index(index)
        async with page.expect_download(timeout=5*1000) as download_info:
            await handle.scroll_into_view_if_needed()
            await handle.click()
        download=await download_info.value
        if filename is None:
            filename=download.suggested_filename
        path=folder_path.joinpath(filename)
        await download.save_as(path=path)
    except:
        async with httpx.AsyncClient() as client:
            response=await client.get(url)
        path=folder_path.joinpath(filename)
        with open(path,'wb') as f:
            f.write(response.content)
    return f'Downloaded {filename} from {url} and saved it to {path}'

@Tool('ExtractContent Tool',params=ExtractContent)
async def extract_content_tool(value:str,context:Context=None):
    '''Extract the information present in a webpage such as text, images, etc'''
    page=await context.get_current_page()
    html=await page.content()
    content=MainContentExtractor.extract(html,output_format=value)
    return f'Extracted Page Content:\n{content}'

@Tool('Tab Tool',params=Tab)
async def tab_tool(mode:Literal['open','close','switch'],tab_index:int=None,context:Context=None):
    '''To open a new tab, close the current tab and switch from current tab to the specified tab'''
    session=await context.get_session()
    if mode=='open':
        page=await session.context.new_page()
        session.current_page=page
        await page.wait_for_load_state('load')
        return f'Opened new tab and switched to it'
    elif mode=='close':
        page=session.current_page
        await page.close()
        pages=session.context.pages
        if tab_index is not None and tab_index>len(pages):
            raise IndexError('Index out of range')
        page=pages[0]
        session.current_page=page
        await page.bring_to_front()
        await page.wait_for_load_state('load')
        return f'Closed current tab and switched to tab 0'
    elif mode=='switch':
        pages=session.context.pages
        if tab_index>len(pages):
            raise IndexError('Index out of range')
        page=pages[tab_index]
        session.current_page=page
        await page.bring_to_front()
        await page.wait_for_load_state('load')
        return f'Switched to tab {tab_index}'
    else:
        raise ValueError('Invalid mode')
    
@Tool('Upload Tool',params=Upload)   
async def upload_tool(index:int,filenames:list[str],context:Context=None):
    '''To upload a file to the webpage'''
    files=[Path(getcwd()).joinpath('./uploads',filename) for filename in filenames]
    page=await context.get_current_page()
    _,handle=await context.get_element_by_index(index)
    async with page.expect_file_chooser() as file_chooser_info:
        await handle.scroll_into_view_if_needed()
        await handle.click()
    file_chooser=await file_chooser_info.value
    if file_chooser.is_multiple():
        await handle.set_input_files(files=files)
    else:
        await handle.set_input_files(files=files[0])
    await page.wait_for_load_state('load')
    return f'Uploaded {filenames} to element {index}'


@Tool('Menu Tool',params=Menu)
async def menu_tool(index:int,labels:list[str],context:Context=None):
    '''To open an element having dropdown menu and select an option from it'''
    _,handle=await context.get_element_by_index(index)
    await handle.scroll_into_view_if_needed()
    label=labels if len(labels)>1 else labels[0]
    await handle.select_option(label=label)
    return f'Opened context menu of element {index} and selected {label}'

@Tool('Form Tool',params=Form)
async def form_tool(tool_names:list[Literal['Click Tool','Type Tool','Upload Tool','Menu Tool']],tool_inputs:list[dict],context:Context=None):
    '''To fill input fields of application form'''
    for tool_name,tool_input in zip(tool_names,tool_inputs):
        if tool_name=='Click Tool':
            await click_tool.async_invoke(index=tool_input['index'],context=context)
        elif tool_name=='Type Tool':
            await type_tool.async_invoke(index=tool_input['index'],text=tool_input.get('text'),context=context)
        elif tool_name=='Upload Tool':
            await upload_tool.async_invoke(index=tool_input['index'],filenames=tool_input.get('filenames'),context=context)
        elif tool_name=='Menu Tool':
            await menu_tool.async_invoke(index=tool_input['index'],labels=tool_input.get('labels'),context=context)
    return f'Filled form with inputs {tool_inputs}'