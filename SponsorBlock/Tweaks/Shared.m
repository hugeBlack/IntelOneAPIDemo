//
//  Shared.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
@import ObjectiveC;
@import Foundation;

/**
 Hook (替换或重载) 一个类的实例方法
 如果该类没有实现该方法但父类有实现，会自动为该类添加一个该方法 (相当于重载/super调用版本)
 
 @param targetClass 要 hook 的目标类
 @param selector 要 hook 的方法选择器
 @param newIMP 新的函数实现
 @param origIMPPtr 传出原始的 IMP (如果不为 NULL)
 @return 是否成功
 */
BOOL JRSwizzleInstanceMethod(Class targetClass, SEL selector, IMP newIMP, IMP *origIMPPtr) {
    if (!targetClass || !selector || !newIMP || !origIMPPtr) {
        return NO;
    }
    
    // 1. 在类层级中查找原始方法（会顺着继承链向上找）
    Method originalMethod = class_getInstanceMethod(targetClass, selector);
    if (!originalMethod) {
        NSLog(@"[%s] Method not found: %@", class_getName(targetClass), NSStringFromSelector(selector));
        return NO;
    }
    
    // 2. 保存原始的 IMP
    *origIMPPtr = method_getImplementation(originalMethod);
    
    // 3. 核心魔法：尝试将原始方法添加到目标类中
    // - 如果目标类已经实现了该方法，class_addMethod 会返回 NO，什么也不做。
    // - 如果目标类没有实现（是继承父类的），这里会将其"复制"一份到目标类的方法列表中。
    // 这样做保证了我们后续的替换只影响当前 targetClass，不会污染父类。
    class_addMethod(targetClass,
                    selector,
                    *origIMPPtr,
                    method_getTypeEncoding(originalMethod));
    
    // 4. 获取现在肯定存在于 targetClass 中的本地方法
    Method localMethod = class_getInstanceMethod(targetClass, selector);
    
    // 5. 将本地方法的实现替换为我们的 newIMP
    method_setImplementation(localMethod, newIMP);
    
    return YES;
}

void swizzle(Class class, SEL originalAction, SEL swizzledAction) {
    method_exchangeImplementations(class_getInstanceMethod(class, originalAction), class_getInstanceMethod(class, swizzledAction));
}
