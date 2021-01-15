-- Sets directional force depending on lumbar state
local function setForce()
    array = get_joint_info(0,2) -- Sets a table named array equal to the array returned by get_joint_info
    stomach = get_body_info(0,3)
    if array["state"] == 2 then -- Checks if lumbar is bending left
        for i = 0, 19 do
            set_body_force(0, i, 2, 0, 0) -- Sets left force of 20 to stomach
        end
        set_joint_state(0,2,3)
    end
    if array["state"] == 1 then -- Checks if lumbar is bending right
        for i = 0, 19 do
            set_body_force(0, i, -2, 0, 0) -- Sets right force of 20 to stomach
        end
        set_joint_state(0,2,3)
    end
end

local function resetForce()
    array = get_joint_info(0,2)
    if array["state"] == (0 or 4) then
        for i = 0, 19 do
            set_body_force(0, i, 0, 0, 0)
        end
    end
end


add_hook("enter_freeze", "apply_force", setForce) -- Executes setForce() once freeze is exited
add_hook("enter_freeze", "reset_force", resetForce)



-- Teleports body back to start
local function findPos() --Find position of joints at start of match
    aX = {} -- Creates 3 arrays to store x, y, and z values
    aY = {}
    aZ = {}
    for i = 0, 20 do -- Loop goes through all 20 joints getting their pos
        bodyPos = get_body_info(0, i)
        aX[i] = (bodyPos.pos.x) -- Fills arrays with x, y, and z positions for each body part
        aY[i] = (bodyPos.pos.y)
        aZ[i] = (bodyPos.pos.z)
    end
end
local function resetPos(key) -- Resets player back to start
    if key == 113 then -- Checks if q is pressed
        for i = 0, 20 do -- Goes through reseting location and rotation
            set_body_rotation(0,i,0,0,0)
            set_body_pos(0,i,aX[i],aY[i],aZ[i])
            set_body_force(0, 1, 0, -20, 0)
        end
    end
end

add_hook("new_game", "find_pos", findPos) --Finds pos at start of game
add_hook("key_down", "reset_pos", resetPos) --Resets pos when q is pressed


local function saveVelo()
    ws = get_world_state()
    if ws.match_frame % 5 == 1 then
        set_joint_state(0, 0, 4)
    end
    if ws.match_frame % 5 == 2 then
        set_joint_state(0, 0, 3)
    end
end

add_hook("draw2d", "save_velo", saveVelo)




